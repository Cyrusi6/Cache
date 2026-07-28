#!/usr/bin/env python3
"""CPU-only canonical-input locker for FPCT-E1 E0-design mechanism shards.

The command tokenizes and aligns only the already-open E0-design population. It
loads no model class, checkpoint or CUDA device.  Its local PT sidecar contains
the exact full-response ``AlignedChatDataset`` feature, prompt-only production
generation inputs, certified runtime parents and answer-query identities.  The
compact JSON manifest freezes hashes/counts and the model-independent row-key
template universe before any E1 model output is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


def _early_main_bootstrap_guard() -> None:
    """Reject non-sealed CLI execution before importing project modules."""

    if __name__ != "__main__":
        return
    try:
        root_index = sys.argv.index("--source-snapshot-root") + 1
        root = Path(sys.argv[root_index])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(
            "formal A5 prepare requires --source-snapshot-root under bootstrap"
        ) from exc
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise RuntimeError("formal A5 prepare source snapshot must be a realpath")
    bootstrap = sys.modules.get("fpct_bootstrap")
    if bootstrap is None:
        try:
            bootstrap = importlib.import_module("fpct_bootstrap")
        except ModuleNotFoundError:
            bootstrap = None
    if bootstrap is None:
        raise RuntimeError(
            "formal A5 prepare requires canonical python -I fpct_bootstrap.py"
        )
    bootstrap_path = Path(str(getattr(bootstrap, "__file__", ""))).absolute()
    expected_bootstrap = root / "script/runtime/fpct_bootstrap.py"
    if (
        bootstrap_path.is_symlink()
        or not bootstrap_path.is_file()
        or bootstrap_path.resolve(strict=True) != expected_bootstrap
        or sys.flags.isolated != 1
        or sys.flags.ignore_environment != 1
    ):
        raise RuntimeError("active fpct bootstrap module is outside the snapshot")
    target = root / "script/experiment/fpct_e1_prepare_input_lock.py"
    bootstrap.require_active(target=target)


_early_main_bootstrap_guard()

import yaml

from script.experiment.fpct_e1_capture_runner import (
    E0_DEV_MANIFEST_RELATIVE,
    TASKS,
    TASK_GROUP_COUNTS,
    load_e0_design_lock,
    sha256_file,
    verify_e0_dev_anchor,
)
from script.experiment.fpct_e1_runtime_backend import (
    GOLD_RESPONSE_TEMPLATE,
    _alignment_sha256,
    _build_aligner,
    _instruction_end,
    _load_task_example,
    _prompt_alignment_details,
    _question_choices,
    _sha256_bytes,
    _to_python,
    _verify_feature_alignment,
    canonical_content_sha256,
    canonical_sample_sha256,
    feature_provenance,
    topology_contract,
)
from script.analysis.fpct_e1_streaming_verify import (
    PARQUET_MANIFEST_NAME,
    PHYSICAL_CHUNK_ROWS,
    SampleRowStream,
    attest_ordered_sample_streams,
    canonical_endpoint_id,
    canonical_json_bytes as streaming_canonical_json_bytes,
    iter_verified_parquet_rows,
    logical_row_count,
    validate_streaming_schema_artifact,
    verify_parquet_stream_artifact,
    write_parquet_stream_artifact,
)
from script.experiment.fpct_e1_a5_prompt_provenance import (
    A5_PROTOCOL_ID,
    EXTRA_CHOICES_ONLY,
    HISTORICAL_EXACT,
    attest_e0_renderer_identity,
    build_census_manifest,
    dual_anchor_record,
    full_question_choices,
    historical_projected_example,
    raw_full_row_sha256,
    summarize_dual_anchor_census,
)


SCHEMA_VERSION = 3
PROTOCOL_ID = "fpct_e1_e0_design_input_lock_v3_actual_runtime_prompt"
EXPECTED_RECEIVER_LAYERS = 28
EXPECTED_QUERY_HEADS = 16
HISTORICAL_MAX_LONG_FORM_ROWS_PER_SAMPLE = 262144
A4_STREAMING_PROTOCOL_ID = (
    "fpct_e1_mechanism_audit_v6_representation_preserving_streaming"
)
# Backwards-compatible symbol for the unchanged representation-only streaming
# sub-contract.  A5 never consumes the old A4 tracked-tree GO receipt.
A4_PROTOCOL_ID = A4_STREAMING_PROTOCOL_ID
A4_STREAMING_SCHEMA_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_streaming_schema.json"
)
A4_HISTORICAL_SYNTHETIC_GATE_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_streaming_synthetic_gate.json"
)
A5_PROMPT_CONTRACT_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5_prompt_contract.json"
)
A5_PROMPT_SCHEMA_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5_prompt_schema.json"
)
A5_SYNTHETIC_GATE_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5_prompt_synthetic_gate.json"
)
SOURCE_SNAPSHOT_RECEIPT_NAME = ".fpct_e1_source_snapshot_receipt.json"
INPUT_LOCK_ROOT_NAME = "input_lock"
RUN_UID_TEMPLATE = "fpct-e1-a5-runtime-prompt-{prefix}-v1"
RUN_ROOT_TEMPLATE = "fpct-e1-a5-{prefix}-v1"
HISTORICAL_EXECUTION_PREFIXES = frozenset(
    ("744a943", "d1698177", "612697df", "07755a40")
)
EXECUTION_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
BLOCKED_RECEIPT_NAME = "A5_INPUT_LOCK_BLOCKED.json"
GO_RECEIPT_NAME = "A5_INPUT_LOCK_GO.json"
RUN_IDENTITY_NAME = "a5_input_lock_execution_identity.json"
PROMPT_CENSUS_NAME = "a5_prompt_census_manifest.json"
PROMPT_CENSUS_RECORDS_NAME = "a5_prompt_census_records.jsonl"

SEALED_PREPARE_SOURCE_CLOSURE = {
    "capture_runner": Path("script/experiment/fpct_e1_capture_runner.py"),
    "runtime_backend": Path("script/experiment/fpct_e1_runtime_backend.py"),
    "streaming_verify": Path("script/analysis/fpct_e1_streaming_verify.py"),
    "streaming_gate": Path("script/analysis/fpct_e1_streaming_synthetic_gate.py"),
    "source_snapshot_lock": Path(
        "script/experiment/fpct_e1_source_snapshot_lock.py"
    ),
}
SEALED_PREPARE_MODULE_NAMES = {
    "capture_runner": "script.experiment.fpct_e1_capture_runner",
    "runtime_backend": "script.experiment.fpct_e1_runtime_backend",
    "streaming_verify": "script.analysis.fpct_e1_streaming_verify",
    "streaming_gate": "script.analysis.fpct_e1_streaming_synthetic_gate",
    "source_snapshot_lock": "script.experiment.fpct_e1_source_snapshot_lock",
}
_TEST_ONLY_SEALED_PREPARE_MARKER = object()


def _canonical_regular_file(path: Path, label: str) -> Path:
    """Return one non-aliased regular file or fail before following a link."""

    absolute = path.absolute()
    try:
        mode = absolute.lstat().st_mode
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} is missing") from exc
    if not stat.S_ISREG(mode) or absolute.is_symlink():
        raise RuntimeError(f"{label} must be a non-symlink regular file")
    if absolute.resolve(strict=True) != absolute:
        raise RuntimeError(f"{label} must use its canonical real path")
    return absolute


def _canonical_real_directory(path: Path, label: str) -> Path:
    """Return one non-aliased directory without accepting a symlink root."""

    absolute = path.absolute()
    try:
        mode = absolute.lstat().st_mode
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} is missing") from exc
    if not stat.S_ISDIR(mode) or absolute.is_symlink():
        raise RuntimeError(f"{label} must be a non-symlink directory")
    if absolute.resolve(strict=True) != absolute:
        raise RuntimeError(f"{label} must use its canonical real path")
    return absolute


def _preflight_producer_file(path: Path, label: str) -> None:
    """Require a canonical parent and an absent/canonical regular final."""

    absolute = path.absolute()
    _canonical_real_directory(absolute.parent, f"{label} parent")
    try:
        mode = absolute.lstat().st_mode
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
    ):
        raise RuntimeError(f"{label} must be absent or a canonical regular file")


def _ensure_producer_directory(path: Path, label: str) -> Path:
    """Preflight parent/path before creating exactly one producer directory."""

    absolute = path.absolute()
    _canonical_real_directory(absolute.parent, f"{label} parent")
    try:
        absolute.lstat()
    except FileNotFoundError:
        absolute.mkdir(parents=False, exist_ok=False)
    return _canonical_real_directory(absolute, label)


def _verified_test_sealed_prepare_sentinel(
    source_snapshot_root: Path, execution_sha: str
) -> tuple[object, str, str]:
    """Mint the explicit unit-test-only bypass for non-I/O prepare tests.

    The production CLI never accepts or constructs this value.  Requiring the
    live pytest marker prevents ordinary library callers from silently opting
    out of the sealed-import contract.
    """

    if "PYTEST_CURRENT_TEST" not in os.environ:
        raise RuntimeError("test-only sealed-prepare sentinel requires pytest")
    return (
        _TEST_ONLY_SEALED_PREPARE_MARKER,
        str(source_snapshot_root.absolute()),
        execution_sha,
    )


def _require_sealed_prepare_execution(
    *,
    repo_root: Path,
    source_snapshot_root: Path,
    execution_sha: str,
    test_sentinel: tuple[object, str, str] | None = None,
) -> dict[str, Any]:
    """Bind this process and its prepare closure to the immutable snapshot.

    This gate runs before tokenizer resolution, dataset loading, rendering, or
    alignment.  Production accepts only ``python -I fpct_bootstrap.py``'s
    same-process sentinel; direct execution and hostile ``PYTHONPATH`` shims
    fail closed.  Unit tests may pass the explicit pytest-only sentinel above.
    """

    snapshot = _canonical_real_directory(
        source_snapshot_root, "sealed prepare source snapshot"
    )
    if repo_root.absolute() != snapshot:
        raise RuntimeError("sealed prepare repo root differs from source snapshot")
    if test_sentinel is not None:
        expected = (
            _TEST_ONLY_SEALED_PREPARE_MARKER,
            str(snapshot),
            execution_sha,
        )
        if "PYTEST_CURRENT_TEST" not in os.environ or test_sentinel != expected:
            raise RuntimeError("invalid test-only sealed-prepare sentinel")
        return {
            "protocol_id": "fpct_e1_a5_test_only_sealed_prepare_v1",
            "repo_root": str(snapshot),
            "execution_sha": execution_sha,
            "pytest_verified_test_sentinel": True,
            "production_eligible": False,
        }

    target = _canonical_regular_file(
        snapshot / "script/experiment/fpct_e1_prepare_input_lock.py",
        "sealed prepare formal target",
    )
    bootstrap_path = _canonical_regular_file(
        snapshot / "script/runtime/fpct_bootstrap.py",
        "sealed prepare bootstrap",
    )
    bootstrap = sys.modules.get("fpct_bootstrap")
    if bootstrap is None:
        try:
            bootstrap = importlib.import_module("fpct_bootstrap")
        except ModuleNotFoundError:
            bootstrap = None
    if bootstrap is None:
        raise RuntimeError(
            "formal A5 prepare requires canonical python -I fpct_bootstrap.py"
        )
    bootstrap_origin = _canonical_regular_file(
        Path(str(getattr(bootstrap, "__file__", ""))),
        "active fpct bootstrap module",
    )
    if bootstrap_origin != bootstrap_path:
        raise RuntimeError("active fpct bootstrap module is outside the snapshot")
    try:
        attestation = bootstrap.require_active(target=target)
    except Exception as exc:
        raise RuntimeError(
            "formal A5 prepare requires the active bootstrap sentinel"
        ) from exc
    if not isinstance(attestation, Mapping):
        raise RuntimeError("sealed prepare bootstrap attestation is malformed")
    if (
        attestation.get("repo_root") != str(snapshot)
        or attestation.get("target") != str(target)
        or attestation.get("protected_data_opens_before_target") != []
        or attestation.get("python", {}).get("flags", {}).get("isolated") != 1
        or attestation.get("python", {}).get("flags", {}).get("ignore_environment")
        != 1
    ):
        raise RuntimeError("sealed prepare bootstrap identity/flags differ")
    git_record = attestation.get("git", {})
    if (
        git_record.get("head") != execution_sha
        or git_record.get("clean") is not True
        or git_record.get("source") != "fpct_e1_source_snapshot_receipt"
    ):
        raise RuntimeError("sealed prepare snapshot Git provenance differs")
    mandatory = attestation.get("mandatory_modules", {})
    formal = mandatory.get("formal_target", {})
    if (
        Path(str(formal.get("file", ""))).absolute() != target
        or formal.get("sha256") != sha256_file(target)
    ):
        raise RuntimeError("sealed prepare formal-target SHA/origin differs")
    for key, record in mandatory.items():
        origin = _canonical_regular_file(
            Path(str(record.get("file", ""))), f"sealed mandatory module {key}"
        )
        try:
            origin.relative_to(snapshot)
        except ValueError as exc:
            raise RuntimeError(
                f"sealed mandatory module {key} is outside the snapshot"
            ) from exc
        if record.get("sha256") != sha256_file(origin):
            raise RuntimeError(f"sealed mandatory module {key} SHA changed")

    closure: dict[str, dict[str, Any]] = {}
    for key, module_name in SEALED_PREPARE_MODULE_NAMES.items():
        module = importlib.import_module(module_name)
        expected_path = _canonical_regular_file(
            snapshot / SEALED_PREPARE_SOURCE_CLOSURE[key],
            f"sealed prepare closure source {key}",
        )
        actual_path = _canonical_regular_file(
            Path(str(getattr(module, "__file__", ""))),
            f"loaded prepare closure module {key}",
        )
        if actual_path != expected_path:
            raise RuntimeError(f"loaded prepare closure module {key} origin differs")
        closure[key] = {
            "module": module_name,
            "path": str(actual_path),
            "sha256": sha256_file(actual_path),
        }

    # A foreign project module is not an allowed dependency even if the named
    # prepare modules above happen to be correct.
    for name, module in tuple(sys.modules.items()):
        if not (name == "rosetta" or name.startswith(("rosetta.", "script."))):
            continue
        raw_origin = getattr(module, "__file__", None)
        if raw_origin is None:
            continue
        origin = _canonical_regular_file(
            Path(str(raw_origin)), f"loaded project module {name}"
        )
        try:
            origin.relative_to(snapshot)
        except ValueError as exc:
            raise RuntimeError(
                f"loaded project module {name} is outside the snapshot"
            ) from exc

    return {
        "protocol_id": "fpct_e1_a5_sealed_prepare_execution_v1",
        "repo_root": str(snapshot),
        "execution_sha": execution_sha,
        "bootstrap": {
            "path": str(bootstrap_path),
            "sha256": sha256_file(bootstrap_path),
        },
        "formal_target": {
            "path": str(target),
            "sha256": sha256_file(target),
        },
        "stable_fingerprint_sha256": attestation.get(
            "stable_fingerprint_sha256"
        ),
        "source_snapshot_tree_sha256": git_record.get("tree_sha256"),
        "source_snapshot_receipt_sha256": git_record.get("receipt_sha256"),
        "mandatory_module_sha256": {
            key: record["sha256"] for key, record in sorted(mandatory.items())
        },
        "prepare_closure": closure,
        "python_isolated": True,
        "python_environment_ignored": True,
        "pytest_verified_test_sentinel": False,
        "production_eligible": True,
    }


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_bytes_no_overwrite(path: Path, payload: bytes) -> str:
    """Publish immutable bytes atomically and verify any concurrent winner.

    ``rename``/``replace`` is deliberately forbidden here because it can replace
    a scientifically different winner.  A same-filesystem hard link supplies
    the no-overwrite atomicity.  If another process wins, its bytes must be
    identical before this invocation is allowed to treat the artifact as
    complete.  A unique O_EXCL temporary makes crash debris unambiguous.
    """

    path = path.absolute()
    _preflight_producer_file(path, "immutable JSON producer final")
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError(
                    f"immutable artifact winner bytes differ: {path}"
                )
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    publish_bytes_no_overwrite(path, canonical_json_bytes(value))


def nested_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(_to_python(value))).hexdigest()


def task_membership_sha256(
    group_contract: Sequence[Mapping[str, Any]],
    semantic_members: Sequence[Mapping[str, Any]],
) -> str:
    """Mechanically bind plan membership to the sidecar's semantic item SHAs."""

    planned: dict[str, str] = {}
    for row in group_contract:
        group = str(row["content_group_sha256"])
        samples = row.get("sample_sha256")
        if not isinstance(samples, list) or len(samples) != 1:
            raise ValueError("input-lock membership requires one sample per group")
        if group in planned:
            raise ValueError("duplicate group in input-lock plan membership")
        planned[group] = str(samples[0])
    observed: dict[str, tuple[str, str]] = {}
    for row in semantic_members:
        group = str(row["content_group_sha256"])
        value = (str(row["sample_sha256"]), str(row["item_semantic_sha256"]))
        if group in observed:
            raise ValueError("duplicate group in input-lock semantic membership")
        if any(len(item) != 64 for item in (group, *value)):
            raise ValueError("input-lock membership contains a malformed SHA")
        observed[group] = value
    if set(planned) != set(observed):
        raise ValueError("plan and semantic input-lock group universes differ")
    if any(planned[group] != observed[group][0] for group in planned):
        raise ValueError("plan and semantic input-lock sample identities differ")
    rows = sorted(
        [sample, group, observed[group][1]] for group, sample in planned.items()
    )
    return nested_sha256(rows)


def answer_query_contract(labels: Sequence[Any]) -> list[dict[str, int]]:
    result = []
    values = [int(value) for value in labels]
    for query in range(max(0, len(values) - 1)):
        token = values[query + 1]
        if token != -100:
            if token < 0:
                raise ValueError("eligible answer target token is negative")
            result.append(
                {
                    "query_position": query,
                    "target_position": query + 1,
                    "target_token_id": token,
                }
            )
    if not result:
        raise ValueError("canonical response contains no eligible answer query")
    return result


def certified_parent_contract(
    topology: Mapping[int, Mapping[str, Any]],
    *,
    source_indices: Sequence[Sequence[Any]] | None = None,
    source_weights: Sequence[Sequence[Any]] | None = None,
) -> list[dict[str, Any]]:
    if (source_indices is None) != (source_weights is None):
        raise ValueError("candidate indices and weights must be supplied together")
    if source_indices is not None and (
        len(source_indices) != len(source_weights)  # type: ignore[arg-type]
        or any(parent >= len(source_indices) for parent in topology)
    ):
        raise ValueError("candidate geometry differs from topology parent universe")
    parents = []
    for parent, record in sorted(topology.items()):
        count = int(record["candidate_count"])
        if count < 2:
            continue
        if not record["within_instruction"]:
            raise ValueError("response parent entered include_response=false input lock")
        output = {
            "parent_position": int(parent),
            "candidate_count": count,
            "prior": [float(value) for value in record["prior"]],
            "topology": record["topology"],
            "statistical_weight": 1.0,
        }
        if source_indices is not None and source_weights is not None:
            indices = [int(value) for value in source_indices[parent]]
            weights = [float(value) for value in source_weights[parent]]
            if len(indices) != 4 or len(weights) != 4:
                raise ValueError("certified candidate geometry must have four slots")
            valid = [index >= 0 and math.isfinite(weight) and weight > 0 for index, weight in zip(indices, weights)]
            legal_indices = [index for index, is_valid in zip(indices, valid) if is_valid]
            legal_weights = [weight for weight, is_valid in zip(weights, valid) if is_valid]
            if (
                len(legal_indices) != count
                or len(set(legal_indices)) != count
                or any(
                    (not is_valid and (index != -1 or weight != 0.0))
                    for index, weight, is_valid in zip(indices, weights, valid)
                )
                or any(
                    not math.isclose(left, right, abs_tol=2e-5, rel_tol=0)
                    for left, right in zip(legal_weights, output["prior"])
                )
            ):
                raise ValueError("certified candidate geometry differs from prior")
            output.update(
                {
                    "candidate_indices": indices,
                    "candidate_valid_mask": valid,
                    "candidate_slot_weights": weights,
                }
            )
        parents.append(output)
    if not parents:
        raise ValueError("certified E0-design sample contains no runtime m>=2 parent")
    return parents


BOUNDARY_REASONS = {
    "zero_length_receiver_interval",
    "duplicate_or_overlap_receiver_offsets",
    "zero_length_source_interval",
    "exact_duplicate_source_offsets",
    "partial_overlap_source_offsets",
    "candidate_without_receiver_intersection",
}
ALIAS_REASONS = {
    "exact_duplicate_source_offsets",
    "partial_overlap_source_offsets",
    "duplicate_or_overlap_receiver_offsets",
}


def _relative_interval(
    offset: Sequence[Any], content_span: Sequence[Any]
) -> tuple[int, int]:
    """Clip one absolute tokenizer offset to message-relative content geometry."""

    start, end = (int(value) for value in offset)
    content_start, content_end = (int(value) for value in content_span)
    return (
        max(0, min(content_end, start) - content_start),
        max(0, min(content_end, end) - content_start),
    )


def _intersection(
    receiver: tuple[int, int], source: tuple[int, int]
) -> tuple[int, int] | None:
    left, right = max(receiver[0], source[0]), min(receiver[1], source[1])
    return (left, right) if right > left else None


def raw_topology_ledger(
    *,
    raw_details: Mapping[str, Any],
    sanitized_details: Mapping[str, Any],
    instruction_end: int,
    task: str,
    sample_sha256: str,
    content_group_sha256: str,
    candidate_window: int,
) -> list[dict[str, Any]]:
    """Conservatively classify every pre-sanitizer prompt parent with raw m>=2."""

    from script.analysis.fpct_e1_mechanism_audit import (
        classify_candidate_topology,
    )

    candidate_window = int(candidate_window)
    if candidate_window < 0:
        raise ValueError("candidate_window must be nonnegative")
    raw = raw_details["soft_alignment"]
    runtime = sanitized_details["soft_alignment"]
    message_sections = [
        section for section in raw_details.get("sections", [])
        if section.get("type") == "message"
    ]
    content_spans_slm = raw_details.get("content_spans_slm", [])
    content_spans_llm = raw_details.get("content_spans_llm", [])
    if not (
        len(message_sections)
        == len(content_spans_slm)
        == len(content_spans_llm)
    ):
        raise ValueError("raw topology requires matched message/content spans")
    rows = []
    for parent in range(min(instruction_end, len(raw["source_indices"]))):
        if not bool(raw_details["message_mask"][parent]):
            continue
        raw_slots = [
            (slot, int(index), float(weight))
            for slot, (index, weight) in enumerate(
                zip(raw["source_indices"][parent], raw["source_weights"][parent])
            )
            if int(index) >= 0 and float(weight) > 0
        ]
        raw_legal = [index for _slot, index, _weight in raw_slots]
        if len(raw_legal) < 2:
            continue
        if len(raw_legal) > 4:
            raise ValueError("raw topology exceeds frozen top-k=4")
        if len(set(raw_legal)) != len(raw_legal):
            raise ValueError("raw topology contains duplicate source indices")
        raw_weights = [weight for _slot, _index, weight in raw_slots]
        if any(not math.isfinite(weight) for weight in raw_weights) or not math.isclose(
            sum(raw_weights), 1.0, abs_tol=2e-5, rel_tol=0
        ):
            raise ValueError("raw topology prior is nonfinite or not normalized")
        runtime_slots = [
            (slot, int(index), float(weight))
            for slot, (index, weight) in enumerate(
                zip(
                    runtime["source_indices"][parent],
                    runtime["source_weights"][parent],
                )
            )
            if int(index) >= 0 and float(weight) > 0
        ]
        runtime_legal = [index for _slot, index, _weight in runtime_slots]
        runtime_weights = [weight for _slot, _index, weight in runtime_slots]
        if not 1 <= len(runtime_legal) <= 4:
            raise ValueError("raw topology runtime support must satisfy 1<=m<=4")
        if len(set(runtime_legal)) != len(runtime_legal):
            raise ValueError("raw topology runtime contains duplicate source indices")
        if any(
            not math.isfinite(weight) for weight in runtime_weights
        ) or not math.isclose(sum(runtime_weights), 1.0, abs_tol=2e-5, rel_tol=0):
            raise ValueError("raw topology runtime prior is nonfinite or not normalized")
        runtime_weight_by_index = {
            index: weight for _slot, index, weight in runtime_slots
        }
        certified = bool(runtime["fpct_certified_mask"][parent])
        reason = str(runtime["fpct_certification_reason"][parent])
        section_matches = [
            (ordinal, section)
            for ordinal, section in enumerate(message_sections)
            if int(section["slm_range"][0]) <= parent < int(section["slm_range"][1])
        ]
        if len(section_matches) != 1:
            raise ValueError("raw topology parent has no unique message section")
        ordinal, section = section_matches[0]
        receiver_offset = tuple(
            int(value) for value in raw_details["slm_offsets"][parent]
        )
        receiver_span = _relative_interval(
            receiver_offset, content_spans_slm[ordinal]
        )
        candidate_records = []
        for slot, index, raw_weight in raw_slots:
            if index >= len(raw_details["llm_offsets"]) or index >= len(
                raw_details["llm_ids"]
            ):
                raise ValueError("raw topology candidate index is out of range")
            source_offset = tuple(
                int(value) for value in raw_details["llm_offsets"][index]
            )
            in_message = (
                int(section["llm_range"][0])
                <= index
                < int(section["llm_range"][1])
            )
            source_span = _relative_interval(
                source_offset, content_spans_llm[ordinal]
            )
            intersection = _intersection(receiver_span, source_span)
            if intersection is not None and in_message:
                origin = "span_overlap"
            elif candidate_window > 0 and in_message:
                origin = "window_neighbor"
            else:
                origin = "fallback_or_unknown"
            complete = (
                intersection is not None
                and intersection[0] == receiver_span[0]
                and intersection[1] == receiver_span[1]
            )
            candidate_records.append(
                {
                    "slot": slot,
                    "source_index": index,
                    "source_token_id": int(raw_details["llm_ids"][index]),
                    "source_offset": list(source_offset),
                    "source_span": list(source_span),
                    "intersection": (
                        list(intersection) if intersection is not None else None
                    ),
                    "intersection_length": (
                        intersection[1] - intersection[0]
                        if intersection is not None
                        else 0
                    ),
                    "origin": origin,
                    "raw_weight": raw_weight,
                    "runtime_retained": index in runtime_weight_by_index,
                    "runtime_weight": runtime_weight_by_index.get(index, 0.0),
                    "complete_receiver_explanation": complete,
                }
            )
        source_spans = [
            tuple(record["source_span"]) for record in candidate_records
        ]
        duplicate_offsets = len(set(source_spans)) != len(source_spans)
        alias = reason in ALIAS_REASONS or duplicate_offsets
        boundary = (
            reason in BOUNDARY_REASONS
            or receiver_span[1] <= receiver_span[0]
            or any(record["origin"] == "fallback_or_unknown" for record in candidate_records)
        )
        all_complete = all(
            record["complete_receiver_explanation"] for record in candidate_records
        )
        independent_complete = (
            all_complete
            and len({record["source_token_id"] for record in candidate_records})
            == len(candidate_records)
            and not duplicate_offsets
        )
        taxonomy = classify_candidate_topology(
            receiver_span,
            source_spans,
            candidate_origins=[record["origin"] for record in candidate_records],
            boundary_or_fallback=boundary,
            independent_competitors=independent_complete,
            certified_partition=certified,
            duplicate_or_overlap_alias=alias,
        )
        if certified:
            if (
                len(runtime_legal) < 2
                or reason != "certified_disjoint_partition"
                or taxonomy != "partition_compositional"
            ):
                raise ValueError("certified raw topology row lost its runtime partition")
        elif len(runtime_legal) > 1:
            raise ValueError("uncertified raw topology row was not slot-0 collapsed")
        span_payload = {
            "receiver_span": receiver_span,
            "candidates": candidate_records,
        }
        rows.append(
            {
                "schema_version": 1,
                "split_role": "e0_design",
                "task": task,
                "sample_sha256": sample_sha256,
                "content_group_sha256": content_group_sha256,
                "parent_position": parent,
                "receiver_token_id": int(raw_details["slm_ids"][parent]),
                "receiver_offset": list(receiver_offset),
                "receiver_span": list(receiver_span),
                "raw_candidate_count": len(raw_legal),
                "runtime_candidate_count": len(runtime_legal),
                "raw_candidate_indices": raw_legal,
                "runtime_candidate_indices": runtime_legal,
                "raw_weights": raw_weights,
                "runtime_weights": runtime_weights,
                "candidates": candidate_records,
                "certified": certified,
                "offset_uncertified": not certified,
                "certification_reason": reason,
                "taxonomy": taxonomy,
                "candidate_window": candidate_window,
                "duplicate_or_overlap_alias": alias,
                "runtime_functional_eligible": bool(
                    certified and len(runtime_legal) >= 2
                ),
                "functional_metrics_present": False,
                "span_geometry_sha256": nested_sha256(span_payload),
            }
        )
    return rows


def row_template_records(
    item: Mapping[str, Any],
    *,
    num_layers: int,
    num_query_heads: int,
    num_kv_heads: int,
) -> list[list[Any]]:
    """Return sorted model-independent suffixes of the executor row key."""

    if num_query_heads % num_kv_heads:
        raise ValueError("receiver Hq is not divisible by Hkv")
    group = num_query_heads // num_kv_heads
    rows: list[list[Any]] = []
    for layer in range(num_layers):
        for query_head in range(num_query_heads):
            kv_head = query_head // group
            for query in item["answer_queries"]:
                for parent in item["certified_parents"]:
                    rows.append(
                        [
                            item["sample_sha256"],
                            item["content_group_sha256"],
                            item["provenance"]["input_sha256"],
                            item["provenance"]["alignment_sha256"],
                            item["provenance"]["labels_sha256"],
                            item["provenance"]["gold_response_sha256"],
                            layer,
                            query_head,
                            kv_head,
                            query["query_position"],
                            query["target_position"],
                            query["target_token_id"],
                            parent["parent_position"],
                            parent["candidate_count"],
                            parent["topology"],
                        ]
                    )
    return sorted(rows, key=canonical_json_bytes)


def expected_long_form_rows(
    *,
    answer_query_count: int,
    certified_parent_count: int,
    num_layers: int,
    num_query_heads: int,
) -> int:
    values = (
        answer_query_count,
        certified_parent_count,
        num_layers,
        num_query_heads,
    )
    if any(isinstance(value, bool) or int(value) != value or value <= 0 for value in values):
        raise ValueError("long-form row factors must be positive integers")
    return logical_row_count(
        num_layers=int(num_layers),
        num_query_heads=int(num_query_heads),
        answer_query_count=int(answer_query_count),
        certified_parent_count=int(certified_parent_count),
    )


def expected_long_form_row_volume(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze exact task volume; quantiles use sorted[ceil(p*n)-1]."""

    if not items:
        raise ValueError("cannot freeze an empty task row-volume contract")
    ordered = sorted(int(item["expected_long_form_rows"]) for item in items)
    if any(value <= 0 for value in ordered):
        raise ValueError("task row-volume contains an invalid per-sample count")
    count = len(ordered)

    def nearest_rank(probability: float) -> int:
        return ordered[math.ceil(probability * count) - 1]

    maximum = ordered[-1]
    argmax = min(
        (item for item in items if int(item["expected_long_form_rows"]) == maximum),
        key=lambda item: (item["sample_sha256"], item["content_group_sha256"]),
    )
    return {
        "formula": "num_layers*num_query_heads*answer_query_count*certified_parent_count",
        "historical_cumulative_sample_ceiling": HISTORICAL_MAX_LONG_FORM_ROWS_PER_SAMPLE,
        "historical_ceiling_operative": False,
        "physical_chunk_rows": PHYSICAL_CHUNK_ROWS,
        "chunk_count": sum(math.ceil(value / PHYSICAL_CHUNK_ROWS) for value in ordered),
        "count": count,
        "sum": sum(ordered),
        "min": ordered[0],
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
        "max": maximum,
        "argmax": {
            "sample_sha256": argmax["sample_sha256"],
            "content_group_sha256": argmax["content_group_sha256"],
        },
        "quantile_method": "nearest_rank",
        "quantile_definition": "sorted[ceil(p*n)-1]",
    }


def row_template_attestation(
    items: Sequence[Mapping[str, Any]],
    *,
    num_layers: int,
    num_query_heads: int,
    num_kv_heads: int,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    previous: bytes | None = None
    for item in sorted(
        items, key=lambda value: (value["sample_sha256"], value["content_group_sha256"])
    ):
        for row in row_template_records(
            item,
            num_layers=num_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
        ):
            payload = canonical_json_bytes(row).rstrip(b"\n")
            if previous is not None and payload <= previous:
                raise ValueError("input lock contains duplicate/nonmonotonic row template")
            previous = payload
            digest.update(payload + b"\n")
            count += 1
    if count == 0:
        raise ValueError("input lock row template universe is empty")
    return {"count": count, "sha256": digest.hexdigest()}


def iter_streaming_row_templates(
    items: Sequence[Mapping[str, Any]],
    *,
    task: str,
    num_layers: int,
    num_query_heads: int,
    num_kv_heads: int,
    schema_sha256: str,
) -> Iterator[SampleRowStream]:
    """Yield compact sample-local ordinal streams in frozen sample order.

    The historical ``row_template_records`` helper remains test-only.  This
    production iterator never constructs or sorts an ``N_s``-sized list.
    """

    endpoint = canonical_endpoint_id(
        0, "input_geometry", "input_geometry", "INPUT_LOCK", task, 0.0
    )
    for item in sorted(
        items, key=lambda value: (value["sample_sha256"], value["content_group_sha256"])
    ):
        if item.get("task") != task:
            raise ValueError("streaming row-template task identity mismatch")
        yield SampleRowStream(
            item,
            num_layers=num_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            endpoint_id=endpoint,
            schema_sha256=schema_sha256,
        )


def streaming_row_template_attestation(
    items: Sequence[Mapping[str, Any]],
    *,
    task: str,
    num_layers: int,
    num_query_heads: int,
    num_kv_heads: int,
    schema_sha256: str,
) -> dict[str, Any]:
    """Attest the exact logical row stream without materializing the table."""

    value = attest_ordered_sample_streams(
        iter_streaming_row_templates(
            items,
            task=task,
            num_layers=num_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            schema_sha256=schema_sha256,
        )
    )
    return {
        "count": value["logical_row_count"],
        "semantic_stream_sha256": value["semantic_stream_sha256"],
        "schema_sha256": value["schema_sha256"],
        "physical_chunk_rows": PHYSICAL_CHUNK_ROWS,
        "sample_count": value["sample_count"],
        "first_sample_sha256": value["first_sample_sha256"],
        "last_sample_sha256": value["last_sample_sha256"],
        "ordinal_order": "sample_sha256,row_ordinal",
    }


def _config_dimensions(receiver_path: Path) -> dict[str, int]:
    config_path = receiver_path / "config.json"
    value = json.loads(config_path.read_text(encoding="utf-8"))
    dimensions = {
        "num_hidden_layers": int(value["num_hidden_layers"]),
        "num_attention_heads": int(value["num_attention_heads"]),
        "num_key_value_heads": int(value["num_key_value_heads"]),
    }
    if dimensions["num_hidden_layers"] != EXPECTED_RECEIVER_LAYERS:
        raise ValueError("receiver layer count differs from frozen 28")
    if dimensions["num_attention_heads"] != EXPECTED_QUERY_HEADS:
        raise ValueError("receiver query-head count differs from frozen 16")
    if dimensions["num_attention_heads"] % dimensions["num_key_value_heads"]:
        raise ValueError("receiver GQA head mapping is invalid")
    return dimensions


def _tokenizer_files(path: Path) -> list[dict[str, Any]]:
    names = (
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    )
    records = []
    for name in names:
        candidate = path / name
        if candidate.is_file():
            records.append(
                {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)}
            )
    if not records:
        raise FileNotFoundError(f"no tokenizer/config files at {path}")
    return records


def runtime_asset_tree(path: Path) -> dict[str, Any]:
    """Freeze every file in one resolved pretrained runtime asset tree."""

    requested = path.absolute()
    if not requested.exists():
        raise FileNotFoundError(requested)
    root_is_symlink = requested.is_symlink()
    root_link_target = os.readlink(requested) if root_is_symlink else None
    resolved = requested.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("runtime asset root is not a directory")
    files: list[dict[str, Any]] = []
    for candidate in sorted(resolved.rglob("*")):
        relative = candidate.relative_to(resolved).as_posix()
        if candidate.is_symlink():
            target_text = os.readlink(candidate)
            target = candidate.resolve(strict=True)
            if target.is_dir():
                raise ValueError(
                    f"runtime asset tree contains unsupported directory symlink: {relative}"
                )
            files.append(
                {
                    "relative_path": relative,
                    "kind": "symlink_file",
                    "symlink_target": target_text,
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
            )
        elif candidate.is_file():
            files.append(
                {
                    "relative_path": relative,
                    "kind": "file",
                    "symlink_target": None,
                    "bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    if not files:
        raise ValueError("runtime asset tree contains no files")
    names = {row["relative_path"] for row in files}
    if "config.json" not in names:
        raise ValueError("runtime asset tree lacks config.json")
    if not any("tokenizer" in Path(name).name for name in names):
        raise ValueError("runtime asset tree lacks tokenizer files")
    if not any(
        name.endswith((".safetensors", ".bin"))
        and "tokenizer" not in Path(name).name
        for name in names
    ):
        raise ValueError("runtime asset tree lacks model weight files")
    portable = {
        "root_kind": "symlink_dir" if root_is_symlink else "directory",
        "root_symlink_target": root_link_target,
        "files": files,
    }
    return {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        **portable,
        "file_count": len(files),
        "bytes": sum(int(row["bytes"]) for row in files),
        "tree_sha256": nested_sha256(portable),
    }


A5_TOKENIZER_ONLY_ALLOWLIST = (
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)
A5_FORBIDDEN_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def tokenizer_runtime_asset_tree(path: Path) -> dict[str, Any]:
    """Hash only prompt/tokenizer/config assets; never open weight/checkpoint files."""

    requested = path.absolute()
    if not requested.exists():
        raise FileNotFoundError(requested)
    root_is_symlink = requested.is_symlink()
    root_link_target = os.readlink(requested) if root_is_symlink else None
    resolved = requested.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("A5 tokenizer asset root is not a directory")
    files: list[dict[str, Any]] = []
    for name in A5_TOKENIZER_ONLY_ALLOWLIST:
        candidate = resolved / name
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            target_text = os.readlink(candidate)
            target = candidate.resolve(strict=True)
            if not target.is_file() or target.is_symlink():
                raise ValueError(
                    f"A5 tokenizer symlink does not resolve to a regular file: {name}"
                )
            kind = "symlink_file"
            digest_path = target
            symlink_target = target_text
        elif stat.S_ISREG(mode):
            kind = "file"
            digest_path = candidate
            symlink_target = None
        else:
            raise ValueError(f"A5 tokenizer asset is not a regular file: {name}")
        files.append(
            {
                "relative_path": name,
                "kind": kind,
                "symlink_target": symlink_target,
                "bytes": digest_path.stat().st_size,
                "sha256": sha256_file(digest_path),
            }
        )
    names = {record["relative_path"] for record in files}
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"}.issubset(
        names
    ):
        raise ValueError("A5 tokenizer-only asset tree lacks required files")
    if any(
        record["relative_path"].endswith(A5_FORBIDDEN_WEIGHT_SUFFIXES)
        for record in files
    ):
        raise AssertionError("A5 tokenizer-only asset walker admitted a weight file")
    portable = {
        "root_kind": "symlink_dir" if root_is_symlink else "directory",
        "root_symlink_target": root_link_target,
        "asset_scope": "tokenizer_config_chat_template_only_no_weights",
        "weight_or_checkpoint_file_opened": False,
        "files": sorted(files, key=lambda record: record["relative_path"]),
    }
    return {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        **portable,
        "file_count": len(files),
        "bytes": sum(int(record["bytes"]) for record in files),
        "tree_sha256": nested_sha256(portable),
    }


def portable_runtime_asset_tree(record: Mapping[str, Any]) -> dict[str, Any]:
    portable = {
        "root_kind": record["root_kind"],
        "root_symlink_target": record.get("root_symlink_target"),
        "files": record["files"],
        "file_count": int(record["file_count"]),
        "bytes": int(record["bytes"]),
        "tree_sha256": record["tree_sha256"],
    }
    if "asset_scope" in record:
        portable["asset_scope"] = record["asset_scope"]
        portable["weight_or_checkpoint_file_opened"] = record.get(
            "weight_or_checkpoint_file_opened"
        )
    return portable


def _generic_asset_tree(path: Path) -> dict[str, Any]:
    """Hash a local input tree without following an unrecorded path alias."""

    requested = path.absolute()
    resolved = requested.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"input asset root is not a directory: {requested}")
    records: list[dict[str, Any]] = []
    for candidate in sorted(resolved.rglob("*")):
        relative = candidate.relative_to(resolved).as_posix()
        if candidate.is_symlink():
            target_text = os.readlink(candidate)
            target = candidate.resolve(strict=True)
            if target.is_dir():
                raise ValueError(
                    f"input asset tree contains a directory symlink: {relative}"
                )
            records.append(
                {
                    "relative_path": relative,
                    "kind": "symlink_file",
                    "symlink_target": target_text,
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
            )
        elif candidate.is_file():
            records.append(
                {
                    "relative_path": relative,
                    "kind": "file",
                    "symlink_target": None,
                    "bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    if not records:
        raise ValueError(f"input asset tree contains no files: {requested}")
    portable = {"files": records}
    return {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "files": records,
        "file_count": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "tree_sha256": nested_sha256(portable),
    }


def input_asset_state(
    *,
    repo_root: Path,
    e0_data_root: Path,
    runtime_assets: Mapping[str, Mapping[str, Any]],
    source_snapshot_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive a before/after digest for every input consulted by this lock."""

    tracked_relatives = (
        A4_STREAMING_SCHEMA_RELATIVE,
        A4_HISTORICAL_SYNTHETIC_GATE_RELATIVE,
        A5_PROMPT_CONTRACT_RELATIVE,
        A5_PROMPT_SCHEMA_RELATIVE,
        A5_SYNTHETIC_GATE_RELATIVE,
        Path("recipe/eval_recipe/fpct_e1/e1_data_split_manifest.json"),
        E0_DEV_MANIFEST_RELATIVE,
        Path("recipe/eval_recipe/fpct_e0/rendered/eval_2026072201_Y_FF_ai2-arc.yaml"),
    )
    tracked = []
    for relative in tracked_relatives:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        tracked.append(
            {
                "relative_path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    portable_runtime = {
        role: portable_runtime_asset_tree(record)
        for role, record in sorted(runtime_assets.items())
    }
    payload = {
        "tracked_source_assets": tracked,
        "e0_data_assets": _generic_asset_tree(e0_data_root),
        "runtime_assets": portable_runtime,
        "source_snapshot_verification": dict(source_snapshot_verification),
    }
    return {**payload, "aggregate_sha256": nested_sha256(payload)}


def _load_a5_prompt_contract(repo_root: Path) -> dict[str, Any]:
    path = repo_root / A5_PROMPT_CONTRACT_RELATIVE
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema_version") != 7
        or contract.get("protocol_id") != A5_PROTOCOL_ID
        or contract.get("amendment_id")
        != "APPROVED_PROSPECTIVE_AMENDMENT_E1_A5_RUNTIME_PROMPT"
        or contract.get("approval", {}).get("selected_option")
        != "ACTUAL_E0_PRODUCTION_RUNTIME_PROMPT"
    ):
        raise ValueError("A5 prompt contract identity/approval is invalid")
    population = contract.get("population", {})
    if (
        population.get("distinct_content_groups") != 326
        or population.get("task_order") != list(TASKS)
        or population.get("task_counts") != TASK_GROUP_COUNTS
        or population.get("start_group_ordinal_one_based") != 1
        or population.get("old_group_161_resume_allowed") is not False
    ):
        raise ValueError("A5 prompt contract population/order changed")
    gate = contract.get("a5_pre_natural_gate", {})
    if gate.get("path") != A5_SYNTHETIC_GATE_RELATIVE.as_posix():
        raise ValueError("A5 prompt contract synthetic-gate path changed")
    return contract


def _verify_all_e0_prompt_configs(
    repo_root: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind every rendered E0 evaluation config, not one convenient cell."""

    index_path = repo_root / "recipe/eval_recipe/fpct_e0/rendered/config_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    oracle = contract["e0_renderer_oracle"]
    if sha256_file(index_path) != oracle["config_index_sha256"]:
        raise ValueError("A5 E0 rendered config index SHA changed")
    if index.get("bundle_sha256") != oracle["rendered_config_bundle_sha256"]:
        raise ValueError("A5 E0 rendered config bundle SHA changed")
    records = [row for row in index.get("records", []) if row.get("kind") == "evaluation"]
    if len(records) != 36:
        raise ValueError("A5 expected exactly 36 E0 evaluation configs")
    expected_cells = {"Y_CC", "Y_CF", "Y_FC", "Y_FF"}
    expected_seeds = {2026072201, 2026072202, 2026072203}
    expected_keys = {
        (seed, cell, task)
        for seed in expected_seeds
        for cell in expected_cells
        for task in TASKS
    }
    observed_keys = {
        (int(row.get("seed", -1)), str(row.get("cell")), str(row.get("task")))
        for row in records
    }
    if observed_keys != expected_keys:
        raise ValueError("A5 E0 evaluation config seed/cell/task universe changed")
    prompt_projection: dict[str, Any] | None = None
    verified: list[dict[str, Any]] = []
    for record in records:
        path = index_path.parent / str(record.get("filename", ""))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError("A5 E0 rendered evaluation config SHA changed")
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        rosetta = value.get("model", {}).get("rosetta_config", {})
        projection = {
            "use_cot": value.get("eval", {}).get("use_cot"),
            "use_template": value.get("eval", {}).get("use_template"),
            "answer_method": value.get("eval", {}).get("answer_method"),
            "do_sample": value.get("model", {})
            .get("generation_config", {})
            .get("do_sample"),
            "enable_thinking": False,
            "base_model": rosetta.get("base_model"),
            "teacher_model": rosetta.get("teacher_model"),
            "alignment_strategy": rosetta.get("alignment_strategy"),
            "soft_alignment_top_k": rosetta.get("soft_alignment_top_k"),
            "soft_alignment_score_mode": rosetta.get("soft_alignment_score_mode"),
            "soft_alignment_min_weight": rosetta.get("soft_alignment_min_weight"),
            "candidate_window": rosetta.get(
                "soft_alignment_candidate_window", 0
            ),
            "soft_alignment_boundary_bonus": rosetta.get(
                "soft_alignment_boundary_bonus"
            ),
            "soft_alignment_boundary_tolerance": rosetta.get(
                "soft_alignment_boundary_tolerance"
            ),
            "fpct_alignment_sanitizer": rosetta.get("fpct_alignment_sanitizer"),
            "include_response": rosetta.get("include_response"),
            "attn_implementation": rosetta.get("attn_implementation"),
        }
        if projection != oracle["canonical_prompt_alignment_config_projection"]:
            raise ValueError("A5 E0 prompt-relevant evaluation config changed")
        if prompt_projection is None:
            prompt_projection = projection
        elif projection != prompt_projection:
            raise ValueError("A5 E0 evaluation configs disagree on prompt semantics")
        verified.append(
            {
                "filename": path.name,
                "sha256": record["sha256"],
                "task": record["task"],
                "seed": record["seed"],
                "cell": record["cell"],
            }
        )
    records_sha256 = nested_sha256(verified)
    if records_sha256 != oracle["ordered_evaluation_config_records_sha256"]:
        raise ValueError("A5 ordered E0 evaluation config record SHA changed")
    return {
        "config_index_sha256": sha256_file(index_path),
        "rendered_config_bundle_sha256": index["bundle_sha256"],
        "evaluation_config_count": len(verified),
        "prompt_projection": prompt_projection,
        "enable_thinking": False,
        "records_sha256": records_sha256,
    }


def _tokenizer_file_record(
    runtime_asset: Mapping[str, Any], relative_path: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in runtime_asset.get("files", [])
        if row.get("relative_path") == relative_path
    ]
    if len(matches) != 1:
        raise ValueError(f"A5 tokenizer asset is missing/ambiguous: {relative_path}")
    return matches[0]


def _tokenizer_bundle_sha256(path: Path) -> str:
    allowed = (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "config.json",
        "generation_config.json",
    )
    files = [path / name for name in allowed if (path / name).is_file()]
    if not files:
        raise ValueError("A5 tokenizer bundle has no frozen files")
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _attest_a5_runtime_assets(
    *,
    contract: Mapping[str, Any],
    runtime_assets: Mapping[str, Mapping[str, Any]],
    receiver: Any,
    sender: Any,
    e0_data_tree_sha256: str,
) -> dict[str, Any]:
    assets = contract["asset_identity"]
    if e0_data_tree_sha256 != assets["materialized_e0_dev_data_tree_sha256"]:
        raise ValueError("A5 materialized E0 development tree SHA changed")
    expected = {
        "receiver": assets["receiver"],
        "sender": assets["sender"],
    }
    required_files = {
        "receiver": {
            "tokenizer.json": expected["receiver"]["tokenizer_json_sha256"],
            "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        },
        "sender": {
            "tokenizer.json": expected["sender"]["tokenizer_json_sha256"],
            "tokenizer.model": expected["sender"]["tokenizer_model_sha256"],
            "tokenizer_config.json": "7b41ba7d0eb91e77914ca3dafde559ea3e19878769b7e68409e89bed5222e77a",
        },
    }
    expected_bundles = {
        "receiver": "d2a315d4ca46d73b53ef973f5eda2561daf90a848e07779fac19ce9761f714be",
        "sender": "5a1a4d8005b2377b26f425fc64322ebcb22e898f8d81c603875eec5f8083809b",
    }
    expected_templates = {
        "receiver": (
            4168,
            "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        ),
        "sender": (
            410,
            "66291cf0045c2425a3a667cf3cbb7af2b11f09e025c02f97245323ab79119362",
        ),
    }
    result: dict[str, Any] = {}
    for role, tokenizer in (("receiver", receiver), ("sender", sender)):
        asset = runtime_assets[role]
        if asset.get("model_id") != expected[role]["name"]:
            raise ValueError(f"A5 {role} tokenizer model ID changed")
        resolved = str(asset.get("resolved_path", ""))
        revision = str(expected[role]["revision"])
        resolved_path = Path(resolved)
        bundle_sha256 = _tokenizer_bundle_sha256(resolved_path)
        if bundle_sha256 != expected_bundles[role]:
            raise ValueError(f"A5 {role} tokenizer bundle SHA changed")
        files = {}
        for relative, digest in required_files[role].items():
            record = _tokenizer_file_record(asset, relative)
            if record.get("sha256") != digest:
                raise ValueError(f"A5 {role} {relative} SHA changed")
            files[relative] = digest
        chat_template = getattr(tokenizer, "chat_template", None)
        if not isinstance(chat_template, str) or not chat_template:
            raise ValueError(f"A5 {role} chat template is missing")
        template_bytes = chat_template.encode("utf-8")
        if (
            len(template_bytes),
            _sha256_bytes(template_bytes),
        ) != expected_templates[role]:
            raise ValueError(f"A5 {role} decoded chat-template bytes changed")
        result[role] = {
            "model_id": asset["model_id"],
            "revision": revision,
            "revision_resolution": "exact_frozen_tokenizer_bundle",
            "tree_sha256": asset["tree_sha256"],
            "tokenizer_bundle_sha256": bundle_sha256,
            "files": files,
            "chat_template_bytes": len(template_bytes),
            "chat_template_sha256": _sha256_bytes(template_bytes),
        }
    return {
        "production_data_tree_exactly_attested": True,
        "materialized_e0_dev_data_tree_sha256": e0_data_tree_sha256,
        "tokenizers": result,
        "enable_thinking": False,
    }


def validate_a5_execution_identity(
    *,
    execution_sha: str,
    source_snapshot_root: Path,
    source_snapshot_receipt: Path,
    run_uid: str,
    run_root: Path,
    output_root: Path,
    output_sidecar_name: str = "e0_design_input_sidecar.pt",
    output_manifest_name: str = "e0_design_input_manifest.json",
    sealed_prepare_execution: Mapping[str, Any] | None = None,
    _test_only_run_parent: Path | None = None,
) -> dict[str, Any]:
    """Bind input lock to one fresh A5 snapshot and never an A4 artifact."""

    if not EXECUTION_SHA_PATTERN.fullmatch(execution_sha):
        raise ValueError("execution_sha must be one lowercase 40-character Git SHA")
    prefix = execution_sha[:8]
    if any(execution_sha.startswith(old) for old in HISTORICAL_EXECUTION_PREFIXES):
        raise ValueError("historical abandoned execution identity is forbidden")
    if run_uid != RUN_UID_TEMPLATE.format(prefix=prefix):
        raise ValueError("run_uid does not match the A5 execution SHA")
    run_root_absolute = run_root.absolute()
    if (
        not run_root_absolute.is_dir()
        or run_root_absolute.is_symlink()
        or run_root_absolute.resolve(strict=True) != run_root_absolute
    ):
        raise ValueError("A5 run root must be an existing non-symlink directory")
    if run_root_absolute.name != RUN_ROOT_TEMPLATE.format(prefix=prefix):
        raise ValueError("run root basename does not match the A5 execution SHA")
    if _test_only_run_parent is not None:
        if (
            "PYTEST_CURRENT_TEST" not in os.environ
            or not isinstance(sealed_prepare_execution, Mapping)
            or sealed_prepare_execution.get("pytest_verified_test_sentinel") is not True
        ):
            raise ValueError("A5 test-only run parent lacks verified pytest sentinel")
        expected_parent = _test_only_run_parent.absolute()
    else:
        expected_parent = Path("/netdisk/lijunsi/fpct-e1")
    expected_run_root = expected_parent / RUN_ROOT_TEMPLATE.format(prefix=prefix)
    if run_root_absolute != expected_run_root:
        raise ValueError("A5 run root is outside the frozen /netdisk execution parent")
    if any(old in str(run_root_absolute) for old in HISTORICAL_EXECUTION_PREFIXES):
        raise ValueError("run root aliases a historical abandoned execution")
    snapshot = source_snapshot_root.absolute()
    if (
        snapshot != run_root_absolute / "source_snapshot"
        or snapshot.is_symlink()
        or snapshot.resolve(strict=True) != snapshot
    ):
        raise ValueError("source snapshot must be the non-aliased A5 run-root snapshot")
    receipt = source_snapshot_receipt.absolute()
    if receipt != snapshot / SOURCE_SNAPSHOT_RECEIPT_NAME:
        raise ValueError("source snapshot receipt path is not canonical")
    output = output_root.absolute()
    if output != run_root_absolute / INPUT_LOCK_ROOT_NAME or output.is_symlink():
        raise ValueError("input-lock output must be the canonical A5 input root")
    downstream_root_names = {"raw", "runtime", "locks", "k8s"}
    allowed_run_entries = {
        "source_snapshot",
        INPUT_LOCK_ROOT_NAME,
        *downstream_root_names,
    }
    unexpected = sorted(
        path.name for path in run_root_absolute.iterdir()
        if path.name not in allowed_run_entries
    )
    if unexpected:
        raise ValueError(f"new A5 run root contains unapproved state: {unexpected}")
    for name in downstream_root_names:
        candidate = run_root_absolute / name
        if candidate.exists() and (
            not candidate.is_dir()
            or candidate.is_symlink()
            or any(candidate.iterdir())
        ):
            raise ValueError(f"pre-input-lock downstream root is not empty: {name}")

    from script.experiment.fpct_e1_source_snapshot_lock import (
        verify_source_snapshot_receipt,
    )

    source_verification = verify_source_snapshot_receipt(
        receipt, snapshot, execution_sha
    )
    if sealed_prepare_execution is not None:
        if (
            sealed_prepare_execution.get("repo_root") != str(snapshot)
            or sealed_prepare_execution.get("execution_sha") != execution_sha
            or sealed_prepare_execution.get("production_eligible") not in {
                True,
                False,
            }
        ):
            raise ValueError("sealed prepare execution attestation is inconsistent")
    identity = {
        "schema_version": 1,
        "protocol_id": "fpct_e1_a5_input_lock_execution_identity_v1",
        "execution_sha": execution_sha,
        "execution_prefix": prefix,
        "run_uid": run_uid,
        "run_root": str(run_root_absolute),
        "source_snapshot_root": str(snapshot),
        "source_snapshot_receipt": {
            "path": str(receipt),
            "bytes": receipt.stat().st_size,
            "file_sha256": sha256_file(receipt),
            "verification": source_verification,
        },
        "input_lock_root": str(output),
        "empty_downstream_roots_verified": sorted(downstream_root_names),
        "historical_execution_resume_allowed": False,
        "historical_artifact_reuse_allowed": False,
    }
    if sealed_prepare_execution is not None:
        identity["sealed_prepare_execution"] = dict(sealed_prepare_execution)
    output = _ensure_producer_directory(output, "A5 input-lock output root")
    identity_path = output / RUN_IDENTITY_NAME
    blocked_path = output / BLOCKED_RECEIPT_NAME
    _preflight_producer_file(identity_path, "A5 execution identity final")
    _preflight_producer_file(blocked_path, "A5 blocked receipt final")
    if blocked_path.exists():
        raise RuntimeError("A5 run is terminally blocked; use a new run identity")
    existing = sorted(path.name for path in output.iterdir())
    if existing and not identity_path.is_file():
        raise ValueError("nonempty A5 input root lacks its immutable identity")
    allowed_resume_entries = {
        RUN_IDENTITY_NAME,
        "input_geometry_samples.parquet",
        "input_geometry_manifest.json",
        "input_geometry_receipt.json",
        "row_templates",
        "input_row_template_chunk_index.json",
        "streaming_input_lock_receipt.json",
        PROMPT_CENSUS_NAME,
        PROMPT_CENSUS_RECORDS_NAME,
        GO_RECEIPT_NAME,
        output_sidecar_name,
        output_manifest_name,
    }
    unexpected_output = sorted(
        name
        for name in existing
        if name not in allowed_resume_entries and not (
            name.startswith(".") and name.endswith(".tmp")
        )
    )
    if unexpected_output:
        raise ValueError(
            f"A5 input root contains unbound artifact state: {unexpected_output}"
        )
    atomic_json(identity_path, identity)
    return {**identity, "identity_sha256": sha256_file(identity_path)}


# Kept only for import compatibility with pre-A5 unit tests.  The validator
# itself accepts exclusively the fresh A5 UID/root patterns and rejects the
# abandoned 07755a40 execution prefix.
validate_a4_execution_identity = validate_a5_execution_identity


def _raw_topology_compact_metadata(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Retain complete parent-level structural identity without raw model data."""

    compact_rows = [
        {
            "parent_position": int(row["parent_position"]),
            "raw_candidate_count": int(row["raw_candidate_count"]),
            "runtime_candidate_count": int(row["runtime_candidate_count"]),
            "raw_candidate_indices": [int(value) for value in row["raw_candidate_indices"]],
            "runtime_candidate_indices": [int(value) for value in row["runtime_candidate_indices"]],
            "raw_weights": [float(value) for value in row["raw_weights"]],
            "runtime_weights": [float(value) for value in row["runtime_weights"]],
            "certified": bool(row["certified"]),
            "certification_reason": str(row["certification_reason"]),
            "taxonomy": str(row["taxonomy"]),
            "span_geometry_sha256": str(row["span_geometry_sha256"]),
        }
        for row in rows
    ]
    if len({row["parent_position"] for row in compact_rows}) != len(compact_rows):
        raise ValueError("raw topology compact metadata has duplicate parents")
    return {
        "row_count": len(compact_rows),
        "rows": compact_rows,
        "compact_rows_sha256": nested_sha256(compact_rows),
        "full_ledger_sha256": nested_sha256(list(rows)),
        "contains_model_output": False,
    }


def expanded_row_absence_proof(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Mechanically prove the torch sidecar contains compact geometry only."""

    forbidden_keys = {"row_ordinal", "logical_row_id", "endpoint_row_id"}
    occurrences: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key in forbidden_keys:
                    occurrences.append(child_path)
                visit(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("compact sidecar has no semantic items")
    for item in items:
        if (
            len(item.get("answer_queries", [])) != int(item.get("Q_s", -1))
            or len(item.get("certified_parents", [])) != int(item.get("P_s", -1))
            or int(item.get("N_s", -1))
            != int(item.get("Q_s", 0))
            * int(item.get("P_s", 0))
            * EXPECTED_RECEIVER_LAYERS
            * EXPECTED_QUERY_HEADS
        ):
            raise ValueError("compact sidecar geometry factors are inconsistent")
    if occurrences:
        raise ValueError("compact sidecar contains expanded logical-row identities")
    return {
        "expanded_logical_rows_present": False,
        "expanded_identity_key_occurrences": 0,
        "compact_item_count": len(items),
        "proof_method": "recursive_key_scan_plus_exact_Q_s_P_s_N_s_factorization",
    }


def _write_compact_geometry_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically publish compact geometry; never overwrite a prior winner."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    path = path.absolute()
    _preflight_producer_file(path, "compact geometry Parquet final")
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_text)
    try:
        # Compact Pass-A has 326 rows, nevertheless its physical buffer is
        # explicitly bounded by the same production chunk maximum.
        writer = None
        try:
            for first in range(0, len(rows), PHYSICAL_CHUNK_ROWS):
                batch = pa.Table.from_pylist(
                    [dict(row) for row in rows[first : first + PHYSICAL_CHUNK_ROWS]]
                )
                if writer is None:
                    writer = pq.ParquetWriter(temporary, batch.schema)
                writer.write_table(batch, row_group_size=PHYSICAL_CHUNK_ROWS)
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise ValueError("compact geometry universe is empty")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or sha256_file(path) != sha256_file(temporary):
                raise RuntimeError("immutable compact geometry winner bytes differ")
            if path.stat().st_size != temporary.stat().st_size:
                raise RuntimeError("immutable compact geometry winner size differs")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _iter_compact_geometry_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Boundedly replay Pass-A rows without ``read_table``/whole-table APIs."""

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=PHYSICAL_CHUNK_ROWS):
        if batch.num_rows > PHYSICAL_CHUNK_ROWS:
            raise RuntimeError("compact geometry replay exceeded physical bound")
        names = batch.schema.names
        for row_index in range(batch.num_rows):
            row = {
                name: batch.column(column_index)[row_index].as_py()
                for column_index, name in enumerate(names)
            }
            canonical_json_bytes(row)
            yield row


def _geometry_row(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 6,
        "protocol_id": A4_PROTOCOL_ID,
        "task": item["task"],
        "sample_sha256": item["sample_sha256"],
        "content_group_sha256": item["content_group_sha256"],
        "item_semantic_sha256": item["item_semantic_sha256"],
        "provenance": dict(item["provenance"]),
        "answer_queries": [dict(value) for value in item["answer_queries"]],
        "certified_parents": [dict(value) for value in item["certified_parents"]],
        "Q_s": int(item["Q_s"]),
        "P_s": int(item["P_s"]),
        "N_s": int(item["N_s"]),
        "answer_query_sequence_sha256": item["answer_query_sequence_sha256"],
        "parent_sequence_sha256": item["parent_sequence_sha256"],
        "raw_topology_compact_sha256": item["raw_topology_compact_sha256"],
        "raw_topology_compact": dict(item["raw_topology_compact"]),
        "expected_chunk_count": int(item["expected_chunk_count"]),
    }


def _verified_geometry_rows(
    *,
    samples_path: Path,
    manifest_path: Path,
    schema_sha256: str,
) -> Iterator[dict[str, Any]]:
    """Verify the immutable Pass-A artifact and yield its sole Pass-B input."""

    if not samples_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("A4 compact geometry artifact is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol_id") != A4_PROTOCOL_ID
        or manifest.get("artifact_type") != "input_geometry_manifest"
        or manifest.get("schema_sha256") != schema_sha256
        or manifest.get("samples_parquet_sha256") != sha256_file(samples_path)
        or manifest.get("samples_parquet_bytes") != samples_path.stat().st_size
    ):
        raise RuntimeError("A4 compact geometry provenance differs")
    digest = hashlib.sha256()
    count = 0
    previous: str | None = None
    for row in _iter_compact_geometry_rows(samples_path):
        sample = str(row.get("sample_sha256"))
        if previous is not None and sample <= previous:
            raise RuntimeError("compact geometry sample order is nonmonotonic")
        if nested_sha256(row.get("raw_topology_compact")) != row.get(
            "raw_topology_compact_sha256"
        ):
            raise RuntimeError("compact geometry raw-topology metadata changed")
        digest.update(canonical_json_bytes(row))
        previous = sample
        count += 1
        yield row
    if (
        count != manifest.get("sample_count")
        or digest.hexdigest() != manifest.get("geometry_semantic_stream_sha256")
    ):
        raise RuntimeError("compact geometry bounded replay differs from manifest")


def _global_row_volume(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed = expected_long_form_row_volume(items)
    return {
        "count": observed["count"],
        "sum": observed["sum"],
        "min": observed["min"],
        "p50": observed["p50"],
        "p95": observed["p95"],
        "max": observed["max"],
        "argmax_sample_sha256": observed["argmax"]["sample_sha256"],
        "argmax_content_group_sha256": observed["argmax"][
            "content_group_sha256"
        ],
    }


def _write_geometry_lock(
    *,
    output_dir: Path,
    items: Sequence[Mapping[str, Any]],
    schema_sha256: str,
    synthetic_gate: Mapping[str, Any],
    input_assets_before: Mapping[str, Any],
    input_assets_after: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Write Pass-A compact geometry and fail closed on disk/inode preflight."""

    output_dir = _canonical_real_directory(output_dir, "A4 input-lock root")
    input_assets_equal = input_assets_before == input_assets_after
    observed_counts = {
        task: sum(item.get("task") == task for item in items) for task in TASKS
    }
    if observed_counts != TASK_GROUP_COUNTS or len(items) != sum(TASK_GROUP_COUNTS.values()):
        raise ValueError("A4 geometry lock population differs from 128/70/128")

    if not input_assets_equal:
        raise RuntimeError("input assets changed between before/after lock hashes")
    samples_path = output_dir / "input_geometry_samples.parquet"
    manifest_path = output_dir / "input_geometry_manifest.json"
    receipt_path = output_dir / "input_geometry_receipt.json"
    for path, label in (
        (samples_path, "compact geometry Parquet final"),
        (manifest_path, "compact geometry manifest final"),
        (receipt_path, "compact geometry receipt final"),
    ):
        _preflight_producer_file(path, label)
    geometry_rows = sorted(
        (_geometry_row(item) for item in items),
        key=lambda row: (row["sample_sha256"], row["content_group_sha256"]),
    )
    if len({row["sample_sha256"] for row in geometry_rows}) != len(geometry_rows):
        raise ValueError("A4 compact geometry contains duplicate samples")
    geometry_digest = hashlib.sha256()
    for row in geometry_rows:
        geometry_digest.update(canonical_json_bytes(row))
    geometry_semantic_sha256 = geometry_digest.hexdigest()
    sample_sequence_sha256 = nested_sha256(
        [
            [
                row["sample_sha256"],
                row["content_group_sha256"],
                row["item_semantic_sha256"],
                row["raw_topology_compact_sha256"],
            ]
            for row in geometry_rows
        ]
    )
    total_rows = sum(int(row["N_s"]) for row in geometry_rows)
    bytes_per_row = synthetic_gate.get("estimated_physical_bytes_per_row")
    if (
        isinstance(bytes_per_row, bool)
        or not isinstance(bytes_per_row, int)
        or bytes_per_row <= 0
    ):
        raise ValueError("synthetic gate lacks a positive physical byte estimate")
    estimated_disk_bytes = total_rows * bytes_per_row
    estimated_inode_count = (
        sum(int(row["expected_chunk_count"]) for row in geometry_rows)
        + 3 * len(geometry_rows)
        + 12
    )
    disk = shutil.disk_usage(output_dir)
    stat = os.statvfs(output_dir)
    free_inodes = int(stat.f_favail)
    disk_ok = int(disk.free) >= estimated_disk_bytes
    inode_ok = free_inodes >= estimated_inode_count
    # Preflight precedes publication, so a failed run can never leave a GO
    # geometry artifact or receipt behind.
    if not disk_ok or not inode_ok:
        failed = []
        if not disk_ok:
            failed.append("disk_preflight")
        if not inode_ok:
            failed.append("inode_preflight")
        raise RuntimeError("A4_PREFLIGHT_FAILED:" + ",".join(failed))

    if receipt_path.exists() and (not samples_path.exists() or not manifest_path.exists()):
        raise RuntimeError("compact geometry receipt exists without its prerequisites")
    if manifest_path.exists() and not samples_path.exists():
        raise RuntimeError("compact geometry manifest exists without its Parquet")
    if samples_path.exists():
        replayed_without_manifest = list(_iter_compact_geometry_rows(samples_path))
        if replayed_without_manifest != geometry_rows:
            raise RuntimeError("crash-resume compact geometry Parquet differs")
    else:
        _write_compact_geometry_parquet(samples_path, geometry_rows)
    manifest = {
        "schema_version": 6,
        "protocol_id": A4_PROTOCOL_ID,
        "artifact_type": "input_geometry_manifest",
        "population": "e0_design",
        "task_group_counts": dict(TASK_GROUP_COUNTS),
        "sample_count": len(items),
        "samples_parquet": samples_path.name,
        "samples_parquet_sha256": sha256_file(samples_path),
        "samples_parquet_bytes": samples_path.stat().st_size,
        "schema_sha256": schema_sha256,
        "sample_sequence_sha256": sample_sequence_sha256,
        "geometry_semantic_stream_sha256": geometry_semantic_sha256,
        "row_volume": _global_row_volume(items),
        "estimated_physical_rows": total_rows,
        "estimated_disk_bytes": estimated_disk_bytes,
        "estimated_inode_count": estimated_inode_count,
        "estimated_inode_formula": (
            "chunk_files + 3*sample_count(sample_dir+chunk_manifest+execution_binding) "
            "+ 12 root/final/temporary artifacts"
        ),
        "disk_preflight_passed": disk_ok,
        "inode_preflight_passed": inode_ok,
        "execution_identity_sha256": execution_identity["identity_sha256"],
        "input_asset_state_sha256": input_assets_after["aggregate_sha256"],
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("crash-resume compact geometry manifest differs")
    else:
        validate_streaming_schema_artifact(manifest)
        atomic_json(manifest_path, manifest)
    replayed = list(
        _verified_geometry_rows(
            samples_path=samples_path,
            manifest_path=manifest_path,
            schema_sha256=schema_sha256,
        )
    )
    if replayed != geometry_rows:
        raise RuntimeError("published compact geometry failed bounded replay")
    receipt = {
        "schema_version": 6,
        "protocol_id": A4_PROTOCOL_ID,
        "artifact_type": "input_geometry_receipt",
        "status": "GO",
        "manifest_sha256": sha256_file(manifest_path),
        "schema_sha256": schema_sha256,
        "sample_count": len(items),
        "expanded_logical_rows_materialized": False,
        "task_counts_exact_128_70_128": True,
        "input_assets_unchanged_during_lock": input_assets_equal,
        "input_asset_state_before_sha256": input_assets_before["aggregate_sha256"],
        "input_asset_state_after_sha256": input_assets_after["aggregate_sha256"],
        "input_asset_state_sha256": input_assets_after["aggregate_sha256"],
        "execution_identity_sha256": execution_identity["identity_sha256"],
        "bounded_geometry_replay_equal": True,
        "pass_b_geometry_source_only": "verified_input_geometry_samples.parquet",
        "e1_pilot_consumed": False,
        "model_or_checkpoint_loaded": False,
        "gpu_or_kubernetes_used": False,
    }
    if receipt_path.exists():
        if json.loads(receipt_path.read_text(encoding="utf-8")) != receipt:
            raise RuntimeError("crash-resume compact geometry receipt differs")
    else:
        validate_streaming_schema_artifact(receipt)
        atomic_json(receipt_path, receipt)
    return {
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
        "samples": {
            "path": str(samples_path),
            "sha256": sha256_file(samples_path),
            "bytes": samples_path.stat().st_size,
        },
        "receipt": {
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
            "bytes": receipt_path.stat().st_size,
        },
        "row_volume": manifest["row_volume"],
        "estimated_disk_bytes": estimated_disk_bytes,
        "estimated_inode_count": estimated_inode_count,
    }


def _write_streaming_template_lock(
    *,
    output_dir: Path,
    dimensions: Mapping[str, int],
    schema_sha256: str,
    geometry_lock: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
    input_assets_before_sha256: str,
    input_assets_after_sha256: str,
    synthetic_gate_path: Path,
    synthetic_gate: Mapping[str, Any],
    verify_only: bool = False,
) -> dict[str, Any]:
    """Pass B from verified immutable Pass-A Parquet, never in-memory items."""

    output_dir = _canonical_real_directory(output_dir, "A4 input-lock root")
    input_assets_equal = (
        isinstance(input_assets_before_sha256, str)
        and len(input_assets_before_sha256) == 64
        and input_assets_before_sha256 == input_assets_after_sha256
    )
    if not input_assets_equal:
        raise RuntimeError("Pass B input-asset before/after SHAs differ")

    index_path = output_dir / "input_row_template_chunk_index.json"
    receipt_path = output_dir / "streaming_input_lock_receipt.json"
    row_template_root = output_dir / "row_templates"
    for path, label in (
        (index_path, "global row-template chunk index"),
        (receipt_path, "global streaming input-lock receipt"),
    ):
        _preflight_producer_file(path, label)
    geometry_samples = Path(str(geometry_lock["samples"]["path"]))
    geometry_manifest = Path(str(geometry_lock["manifest"]["path"]))
    if geometry_samples.parent != output_dir or geometry_manifest.parent != output_dir:
        raise RuntimeError("Pass B geometry lock path is not canonical")
    _canonical_regular_file(geometry_samples, "Pass B geometry Parquet")
    _canonical_regular_file(geometry_manifest, "Pass B geometry manifest")
    if (
        geometry_lock["samples"]["sha256"] != sha256_file(geometry_samples)
        or geometry_lock["manifest"]["sha256"] != sha256_file(geometry_manifest)
    ):
        raise RuntimeError("Pass B geometry lock SHA is not immutable")
    geometry_rows = list(
        _verified_geometry_rows(
            samples_path=geometry_samples,
            manifest_path=geometry_manifest,
            schema_sha256=schema_sha256,
        )
    )
    if verify_only:
        _canonical_real_directory(
            row_template_root, "completed row-template root"
        )
        _canonical_regular_file(index_path, "completed global chunk index")
        _canonical_regular_file(receipt_path, "completed streaming receipt")
    else:
        row_template_root = _ensure_producer_directory(
            row_template_root, "row-template producer root"
        )
    if receipt_path.exists() and not index_path.exists():
        raise RuntimeError("A4 streaming receipt exists without its chunk index")
    records: list[dict[str, Any]] = []
    emitted = 0
    generated_digest = hashlib.sha256()
    replay_digest = hashlib.sha256()
    expected = 0
    observed_counts = {task: 0 for task in TASKS}
    expected_sample_roots = {str(item["sample_sha256"]) for item in geometry_rows}
    unexpected_roots = [
        entry.name
        for entry in row_template_root.iterdir()
        if entry.name not in expected_sample_roots
    ]
    if unexpected_roots:
        raise RuntimeError("row-template root contains an unreferenced sample artifact")

    # Scan every pre-existing sample root and every deterministic final/temp
    # before the first binding/chunk is written.  A hostile later sample can
    # therefore never be discovered only after earlier samples were published.
    for item in geometry_rows:
        candidate_root = row_template_root / str(item["sample_sha256"])
        try:
            candidate_root.lstat()
        except FileNotFoundError:
            continue
        candidate_root = _canonical_real_directory(
            candidate_root, "pre-existing sample stream producer root"
        )
        candidate_stream = SampleRowStream(
            item,
            num_layers=int(dimensions["num_hidden_layers"]),
            num_query_heads=int(dimensions["num_attention_heads"]),
            num_kv_heads=int(dimensions["num_key_value_heads"]),
            endpoint_id=canonical_endpoint_id(
                0, "input_geometry", "input_geometry", "INPUT_LOCK", item["task"], 0.0
            ),
            schema_sha256=schema_sha256,
        )
        allowed_names = {"execution_binding.json", PARQUET_MANIFEST_NAME}
        for chunk_index in range(
            math.ceil(candidate_stream.row_count / PHYSICAL_CHUNK_ROWS)
        ):
            allowed_names.add(f"chunk_{chunk_index:08d}.parquet")
        for entry in candidate_root.iterdir():
            if entry.name not in allowed_names and not (
                entry.name == f".{PARQUET_MANIFEST_NAME}.tmp"
                or re.fullmatch(r"\.chunk_[0-9]{8}\.parquet\.tmp", entry.name)
            ):
                raise RuntimeError("sample stream contains an unreferenced artifact")
            _preflight_producer_file(entry, "pre-existing sample producer artifact")

    for item in geometry_rows:
        if item.get("task") not in observed_counts:
            raise ValueError("A4 geometry contains an unexpected task")
        observed_counts[item["task"]] += 1
        stream = SampleRowStream(
            item,
            num_layers=int(dimensions["num_hidden_layers"]),
            num_query_heads=int(dimensions["num_attention_heads"]),
            num_kv_heads=int(dimensions["num_key_value_heads"]),
            endpoint_id=canonical_endpoint_id(
                0,
                "input_geometry",
                "input_geometry",
                "INPUT_LOCK",
                item["task"],
                0.0,
            ),
            schema_sha256=schema_sha256,
        )
        if stream.row_count != int(item["N_s"]):
            raise ValueError("compact geometry N_s differs from ordinal stream")
        sample_root = output_dir / "row_templates" / item["sample_sha256"]
        if verify_only:
            sample_root = _canonical_real_directory(
                sample_root, "completed sample stream root"
            )
        else:
            sample_root = _ensure_producer_directory(
                sample_root, "sample stream producer root"
            )
        binding_path = sample_root / "execution_binding.json"
        binding = {
            "schema_version": 1,
            "protocol_id": "fpct_e1_a4_sample_stream_execution_binding_v1",
            "execution_identity_sha256": execution_identity["identity_sha256"],
            "sample_sha256": item["sample_sha256"],
            "item_semantic_sha256": item["item_semantic_sha256"],
            "historical_artifact_reuse_allowed": False,
        }
        manifest_path = sample_root / PARQUET_MANIFEST_NAME
        if not verify_only:
            producer_paths = [
                binding_path,
                manifest_path,
                sample_root / f".{PARQUET_MANIFEST_NAME}.tmp",
            ]
            for chunk_index in range(
                math.ceil(stream.row_count / PHYSICAL_CHUNK_ROWS)
            ):
                chunk_name = f"chunk_{chunk_index:08d}.parquet"
                producer_paths.extend(
                    (sample_root / chunk_name, sample_root / f".{chunk_name}.tmp")
                )
            for path in producer_paths:
                _preflight_producer_file(path, "sample stream producer artifact")
        if verify_only:
            _canonical_regular_file(
                binding_path, "completed sample execution binding"
            )
            if not binding_path.is_file():
                raise RuntimeError("completed sample stream lacks execution binding")
            if json.loads(binding_path.read_text(encoding="utf-8")) != binding:
                raise RuntimeError("completed sample execution binding changed")
        else:
            if sample_root.exists() and not binding_path.exists() and any(
                sample_root.iterdir()
            ):
                raise RuntimeError("sample stream has unbound pre-existing artifacts")
            atomic_json(binding_path, binding)
        if verify_only:
            _canonical_regular_file(
                manifest_path, "completed sample chunk manifest"
            )
            if not manifest_path.is_file():
                raise RuntimeError("completed sample stream lacks chunk manifest")
            result = verify_parquet_stream_artifact(manifest_path, stream)
        else:
            try:
                result = write_parquet_stream_artifact(sample_root, stream)
            except FileExistsError:
                # A concurrent immutable winner is acceptable only after complete
                # semantic/physical verification against the frozen stream.
                result = verify_parquet_stream_artifact(manifest_path, stream)
        verified = verify_parquet_stream_artifact(manifest_path, stream)
        if result != verified:
            raise ValueError("A4 Parquet write/replay receipts differ")
        physical_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sample_files = {
            binding_path.name,
            manifest_path.name,
            *{
                str(record["relative_path"])
                for record in physical_manifest.get("chunks", [])
            },
        }
        observed_sample_files = {
            path.name for path in sample_root.iterdir() if path.is_file()
        }
        observed_sample_dirs = [path.name for path in sample_root.iterdir() if path.is_dir()]
        observed_sample_symlinks = [
            path.name for path in sample_root.iterdir() if path.is_symlink()
        ]
        if (
            observed_sample_files != expected_sample_files
            or observed_sample_dirs
            or observed_sample_symlinks
        ):
            raise RuntimeError("sample stream contains an unreferenced artifact")
        for row in stream.iter_rows():
            generated_digest.update(streaming_canonical_json_bytes(row))
        replay_count = 0
        for row in iter_verified_parquet_rows(manifest_path, stream):
            replay_digest.update(streaming_canonical_json_bytes(row))
            replay_count += 1
        if replay_count != stream.row_count:
            raise ValueError("A4 physical replay count differs from N_s")
        expected += stream.row_count
        emitted += int(verified["emitted_logical_rows"])
        records.append(
            {
                "task": item["task"],
                "sample_sha256": item["sample_sha256"],
                "content_group_sha256": item["content_group_sha256"],
                "manifest_relative_path": str(manifest_path.relative_to(output_dir)),
                "manifest_sha256": sha256_file(manifest_path),
                "manifest_bytes": manifest_path.stat().st_size,
                "expected_logical_rows": stream.row_count,
                "emitted_logical_rows": verified["emitted_logical_rows"],
                "chunk_count": verified["chunk_count"],
                "semantic_stream_sha256": verified["semantic_stream_sha256"],
                "item_semantic_sha256": item["item_semantic_sha256"],
                "raw_topology_compact_sha256": item[
                    "raw_topology_compact_sha256"
                ],
                "execution_binding_relative_path": str(
                    binding_path.relative_to(output_dir)
                ),
                "execution_binding_sha256": sha256_file(binding_path),
            }
        )
    if observed_counts != TASK_GROUP_COUNTS or len(records) != sum(
        TASK_GROUP_COUNTS.values()
    ):
        raise ValueError("A4 streaming lock population differs from 128/70/128")
    row_template_root = output_dir / "row_templates"
    observed_sample_roots = {
        path.name
        for path in row_template_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    expected_sample_roots = {record["sample_sha256"] for record in records}
    non_directory_entries = [
        path.name
        for path in row_template_root.iterdir()
        if not path.is_dir() or path.is_symlink()
    ]
    if observed_sample_roots != expected_sample_roots or non_directory_entries:
        raise RuntimeError("row-template root contains an unreferenced sample artifact")
    semantic_sha256 = generated_digest.hexdigest()
    replay_sha256 = replay_digest.hexdigest()
    if expected != emitted or semantic_sha256 != replay_sha256:
        raise RuntimeError("A4 logical row emission/replay is not representation-preserving")
    index = {
        "schema_version": 6,
        "protocol_id": A4_PROTOCOL_ID,
        "artifact_type": "chunk_manifest_index",
        "population": "e0_design",
        "physical_chunk_rows": PHYSICAL_CHUNK_ROWS,
        "sample_count": len(records),
        "expected_logical_rows": expected,
        "emitted_logical_rows": emitted,
        "semantic_stream_sha256": semantic_sha256,
        "schema_sha256": schema_sha256,
        "records": records,
        "pass_b_geometry_source": {
            "path": str(geometry_samples),
            "sha256": sha256_file(geometry_samples),
            "manifest_path": str(geometry_manifest),
            "manifest_sha256": sha256_file(geometry_manifest),
            "sole_geometry_input": True,
        },
        "execution_identity_sha256": execution_identity["identity_sha256"],
    }
    if index_path.exists():
        existing_index = json.loads(index_path.read_text(encoding="utf-8"))
        if existing_index != index:
            raise RuntimeError("existing A4 chunk index differs from deterministic replay")
    elif verify_only:
        raise RuntimeError("completed A4 streaming lock lacks its chunk index")
    else:
        validate_streaming_schema_artifact(index)
        atomic_json(index_path, index)
    synthetic_checks = (
        synthetic_gate.get("streaming_contract_checks", {})
        if synthetic_gate.get("protocol_id") == A5_PROTOCOL_ID
        else synthetic_gate.get("checks", {})
    )
    required_synthetic = (
        "row_key_reference_equivalence",
        "weights_reference_equivalence",
        "topology_reference_equivalence",
        "chunk_partition_semantic_equivalence",
        "aggregate_partition_equivalence",
        "bounded_peak_rss",
        "atomic_no_overwrite",
        "crash_resume_equivalence",
    )
    if any(synthetic_checks.get(name) is not True for name in required_synthetic):
        raise ValueError("A4 tracked synthetic gate lacks a required GO check")
    receipt = {
        "schema_version": 6,
        "protocol_id": A4_PROTOCOL_ID,
        "artifact_type": "streaming_input_lock_receipt",
        "status": "GO",
        "protocol_and_schema_versioned": True,
        "old_attempt_artifact_reused": False,
        "e1_pilot_consumed": False,
        "model_or_checkpoint_loaded": False,
        "gpu_or_kubernetes_used": False,
        "expected_logical_rows_eq_emitted": True,
        "ordinal_first_eq_zero": True,
        "ordinal_last_eq_expected_minus_one": True,
        "ordinal_ranges_contiguous": True,
        "missing_rows": 0,
        "duplicate_rows": 0,
        "overlapping_chunks": 0,
        "row_key_reference_equivalence": True,
        "weights_reference_equivalence": True,
        "topology_reference_equivalence": True,
        "semantic_stream_replay_equal": True,
        "chunk_partition_semantic_equivalence": True,
        "aggregate_partition_equivalence": True,
        "bounded_peak_rss": True,
        "whole_table_materialization_detected": False,
        "atomic_no_overwrite": True,
        "crash_resume_equivalence": True,
        "raw_topology_complete": True,
        "task_counts_exact_128_70_128": True,
        "input_assets_unchanged_during_lock": input_assets_equal,
        "input_asset_state_before_sha256": input_assets_before_sha256,
        "input_asset_state_after_sha256": input_assets_after_sha256,
        "pass_b_consumed_verified_geometry_only": True,
        "expanded_logical_rows_in_compact_sidecar": False,
        "execution_identity_sha256": execution_identity["identity_sha256"],
        "geometry_manifest_sha256": geometry_lock["manifest"]["sha256"],
        "chunk_manifest_index_sha256": sha256_file(index_path),
        "semantic_stream_sha256": semantic_sha256,
        "schema_sha256": schema_sha256,
        "rss_lock_sha256": sha256_file(synthetic_gate_path),
    }
    if receipt_path.exists():
        existing_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing_receipt != receipt:
            raise RuntimeError("existing A4 streaming receipt changed")
    elif verify_only:
        raise RuntimeError("completed A4 streaming lock lacks its receipt")
    else:
        validate_streaming_schema_artifact(receipt)
        atomic_json(receipt_path, receipt)
    return {
        "index": {
            "path": str(index_path),
            "sha256": sha256_file(index_path),
            "bytes": index_path.stat().st_size,
        },
        "receipt": {
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
            "bytes": receipt_path.stat().st_size,
        },
        "expected_logical_rows": expected,
        "emitted_logical_rows": emitted,
        "semantic_stream_sha256": semantic_sha256,
    }


def _verify_completed_file_record(
    record: Mapping[str, Any], expected_path: Path, label: str
) -> None:
    """Fail closed unless one recorded immutable file is still byte-identical."""

    if not isinstance(record, Mapping):
        raise RuntimeError(f"completed {label} is missing")
    expected_path = _canonical_regular_file(expected_path, f"completed {label}")
    recorded_path = Path(str(record.get("path", ""))).absolute()
    if (
        recorded_path != expected_path.absolute()
        or record.get("sha256") != sha256_file(expected_path)
        or record.get("bytes") != expected_path.stat().st_size
    ):
        raise RuntimeError(f"completed {label} path/SHA/size changed")


def _verify_a5_census_artifacts(
    *,
    repo_root: Path,
    output_root: Path,
    completed: Mapping[str, Any],
    locked_payload: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Strictly reopen every canonical JSONL row and independently reduce it."""

    from script.analysis.fpct_e1_a5_prompt_gate import validate_a5_schema_artifact

    records_path = _canonical_regular_file(
        output_root / PROMPT_CENSUS_RECORDS_NAME, "A5 census JSONL"
    )
    manifest_path = _canonical_regular_file(
        output_root / PROMPT_CENSUS_NAME, "A5 census manifest"
    )
    records: list[dict[str, Any]] = []
    with records_path.open("rb") as handle:
        for ordinal, line in enumerate(handle, start=1):
            if not line.endswith(b"\n") or line in {b"\n", b"\r\n"}:
                raise RuntimeError("A5 census JSONL line framing changed")
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("A5 census JSONL is malformed") from error
            if not isinstance(value, dict) or canonical_json_bytes(value) != line:
                raise RuntimeError("A5 census JSONL is not canonical one-row-per-line")
            validate_a5_schema_artifact(value, repo_root=repo_root)
            records.append(value)
    expected_population = sum(TASK_GROUP_COUNTS.values())
    if len(records) != expected_population:
        raise RuntimeError("A5 census JSONL row count differs from frozen population")
    census_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_a5_schema_artifact(census_manifest, repo_root=repo_root)
    rebuilt = build_census_manifest(
        records,
        execution_sha=str(execution_identity["execution_sha"]),
        run_uid=str(execution_identity["run_uid"]),
        record_artifact={
            "relative_path": PROMPT_CENSUS_RECORDS_NAME,
            "sha256": sha256_file(records_path),
            "bytes": records_path.stat().st_size,
            "row_count": len(records),
        },
        expected_task_counts=TASK_GROUP_COUNTS,
    )
    if rebuilt != census_manifest:
        raise RuntimeError("A5 census manifest does not independently recompute")
    census_binding = completed.get("census", {})
    for name, path in (("records", records_path), ("manifest", manifest_path)):
        record = census_binding.get(name, {})
        if (
            Path(str(record.get("path", ""))).absolute() != path.absolute()
            or record.get("sha256") != sha256_file(path)
            or record.get("bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"A5 top-level census {name} binding changed")
    if (
        census_binding.get("record_count") != len(records)
        or census_binding.get("canonical_semantic_stream_sha256")
        != census_manifest["canonical_semantic_stream_sha256"]
    ):
        raise RuntimeError("A5 top-level census semantic binding changed")
    items = locked_payload.get("items")
    if not isinstance(items, list) or len(items) != len(records):
        raise RuntimeError("A5 census/compact-sidecar population differs")
    item_by_key = {
        (
            str(item.get("task")),
            str(item.get("content_group_sha256")),
            str(item.get("sample_sha256")),
        ): item
        for item in items
    }
    if len(item_by_key) != len(items):
        raise RuntimeError("A5 compact sidecar has duplicate census identity")
    for record in records:
        key = (
            record["task"],
            record["content_group_sha256"],
            record["sample_key_sha256"],
        )
        item = item_by_key.get(key)
        if item is None or any(
            item.get(item_name) != record[record_name]
            for item_name, record_name in (
                ("production_rendered_prompt_sha256", "production_rendered_prompt_sha256"),
                ("historical_rendered_prompt_sha256", "historical_rendered_prompt_sha256"),
                ("production_alignment_sha256", "production_alignment_sha256"),
                ("historical_alignment_sha256", "historical_alignment_sha256"),
                ("raw_full_row_sha256", "raw_full_row_sha256"),
                ("prompt_relation", "prompt_relation"),
                ("choice_difference_only", "choice_difference_only"),
            )
        ):
            raise RuntimeError("A5 census row differs from compact sidecar semantics")
    return census_manifest, records


def _clean_owned_completed_crash_debris(
    output_root: Path, *, output_sidecar_name: str, output_manifest_name: str
) -> None:
    """Remove only deterministic producer-owned temporaries from a GO root.

    This is debris collection, never artifact repair: final files are neither
    created nor replaced.  Unknown dot-files and symlinks remain visible to the
    later exact-set checks and therefore fail closed.
    """

    root_finals = {
        RUN_IDENTITY_NAME,
        "input_geometry_samples.parquet",
        "input_geometry_manifest.json",
        "input_geometry_receipt.json",
        "input_row_template_chunk_index.json",
        "streaming_input_lock_receipt.json",
        PROMPT_CENSUS_RECORDS_NAME,
        PROMPT_CENSUS_NAME,
        GO_RECEIPT_NAME,
        output_sidecar_name,
        output_manifest_name,
    }

    def owned(name: str, finals: Sequence[str]) -> bool:
        return any(
            name == f".{final}.tmp"
            or re.fullmatch(
                rf"\.{re.escape(final)}\.[A-Za-z0-9_-]{{6,32}}\.tmp", name
            )
            is not None
            for final in finals
        )

    cleaned_roots: set[Path] = set()
    for path in output_root.iterdir():
        if owned(path.name, tuple(root_finals)):
            if not path.is_file() or path.is_symlink():
                raise RuntimeError("owned crash-debris path is not a regular file")
            path.unlink()
            cleaned_roots.add(output_root)
    row_templates = output_root / "row_templates"
    if row_templates.is_dir() and not row_templates.is_symlink():
        for sample_root in row_templates.iterdir():
            if (
                not sample_root.is_dir()
                or sample_root.is_symlink()
                or re.fullmatch(r"[0-9a-f]{64}", sample_root.name) is None
            ):
                continue
            sample_finals = {
                "execution_binding.json",
                PARQUET_MANIFEST_NAME,
                *{
                    path.name
                    for path in sample_root.glob("chunk_*.parquet")
                    if path.is_file() and not path.is_symlink()
                },
            }
            for path in sample_root.iterdir():
                if owned(path.name, tuple(sample_finals)) or re.fullmatch(
                    r"\.chunk_[0-9]{6}\.parquet\.tmp", path.name
                ):
                    if not path.is_file() or path.is_symlink():
                        raise RuntimeError(
                            "owned sample crash-debris path is not a regular file"
                        )
                    path.unlink()
                    cleaned_roots.add(sample_root)
    for directory in sorted(cleaned_roots):
        _fsync_directory(directory)


def _verify_completed_input_lock(
    *,
    repo_root: Path,
    e0_data_root: Path,
    output_sidecar: Path,
    output_manifest: Path,
    execution_identity: Mapping[str, Any],
    expect_go_receipt: bool = True,
) -> dict[str, Any]:
    """Deeply replay an existing GO lock without creating or repairing artifacts.

    A completed top-level manifest is only a claim.  Resume accepts that claim
    after independently re-verifying every producer/consumer edge: source and
    execution identity, tracked synthetic gate, current input assets, compact
    geometry, all per-sample Parquet chunks, the global template index/receipt,
    and the compact sidecar.  No missing file is reconstructed on this path.
    """

    import torch
    from script.analysis.fpct_e1_a5_prompt_gate import (
        validate_a5_schema_artifact,
        verify_a5_gate,
    )

    output_root = output_manifest.absolute().parent
    _clean_owned_completed_crash_debris(
        output_root,
        output_sidecar_name=output_sidecar.name,
        output_manifest_name=output_manifest.name,
    )
    if (
        not output_manifest.is_file()
        or output_manifest.is_symlink()
        or not output_sidecar.is_file()
        or output_sidecar.is_symlink()
    ):
        raise RuntimeError("completed input lock lacks its manifest or sidecar")
    completed = json.loads(output_manifest.read_text(encoding="utf-8"))
    streaming_schema_path = (repo_root / A4_STREAMING_SCHEMA_RELATIVE).absolute()
    synthetic_gate_path = (repo_root / A5_SYNTHETIC_GATE_RELATIVE).absolute()
    if (
        not streaming_schema_path.is_file()
        or streaming_schema_path.is_symlink()
        or not synthetic_gate_path.is_file()
        or synthetic_gate_path.is_symlink()
    ):
        raise FileNotFoundError("A5 streaming schema/synthetic gate is unavailable")
    streaming_schema_sha256 = sha256_file(streaming_schema_path)
    validate_a5_schema_artifact(completed, repo_root=repo_root)

    if (
        completed.get("schema_version") != 7
        or completed.get("protocol_id") != A5_PROTOCOL_ID
        or completed.get("artifact_type") != "a5_input_lock_manifest"
        or completed.get("status")
        != "A5_INPUT_LOCK_GO_NO_MODEL_OUTPUT"
        or completed.get("execution", {}).get("execution_sha")
        != execution_identity.get("execution_sha")
        or completed.get("execution", {}).get("run_uid")
        != execution_identity.get("run_uid")
        or completed.get("execution", {}).get("run_root")
        != execution_identity.get("run_root")
    ):
        raise RuntimeError("completed A5 input lock identity/status changed")

    identity_path = output_root / RUN_IDENTITY_NAME
    identity_payload = dict(execution_identity)
    identity_sha256 = identity_payload.pop("identity_sha256", None)
    if (
        not identity_path.is_file()
        or identity_path.is_symlink()
        or identity_sha256 != sha256_file(identity_path)
        or json.loads(identity_path.read_text(encoding="utf-8")) != identity_payload
    ):
        raise RuntimeError("completed A5 execution identity changed")
    source_receipt = execution_identity.get("source_snapshot_receipt", {})
    source_receipt_path = Path(str(source_receipt.get("path", ""))).absolute()
    if (
        not source_receipt_path.is_file()
        or source_receipt_path.is_symlink()
        or source_receipt_path
        != Path(str(execution_identity.get("source_snapshot_root", ""))).absolute()
        / SOURCE_SNAPSHOT_RECEIPT_NAME
        or source_receipt.get("file_sha256") != sha256_file(source_receipt_path)
        or source_receipt.get("bytes") != source_receipt_path.stat().st_size
    ):
        raise RuntimeError("completed source-snapshot receipt binding changed")
    completed_execution = completed.get("execution", {})
    if (
        completed_execution.get("source_snapshot_receipt_sha256")
        != source_receipt["file_sha256"]
        or completed_execution.get("source_snapshot_tree_sha256")
        != source_receipt.get("verification", {}).get(
            "mounted_tree_canonical_sha256"
        )
    ):
        raise RuntimeError("completed A5 source-snapshot execution binding changed")

    synthetic_gate = verify_a5_gate(synthetic_gate_path, repo_root=repo_root)
    if (
        synthetic_gate.get("protocol_id") != A5_PROTOCOL_ID
        or synthetic_gate.get("status") != "GO_PRE_NATURAL_SYNTHETIC_HARD_GATE"
        or synthetic_gate.get("natural_e0_design_accessed") is not False
    ):
        raise RuntimeError("completed lock synthetic gate/source binding changed")
    sidecar_record = completed.get("sidecar", {})
    if (
        Path(str(sidecar_record.get("path", ""))).absolute()
        != output_sidecar.absolute()
        or sidecar_record.get("file_sha256") != sha256_file(output_sidecar)
        or sidecar_record.get("bytes") != output_sidecar.stat().st_size
    ):
        raise RuntimeError("completed compact sidecar path/SHA/size changed")
    locked_payload = torch.load(output_sidecar, map_location="cpu", weights_only=False)
    if not isinstance(locked_payload, Mapping):
        raise RuntimeError("completed compact sidecar is not a mapping")
    streaming_contract = locked_payload.get("streaming_contract", {})
    synthetic_record = streaming_contract.get("synthetic_gate", {})
    if (
        Path(str(synthetic_record.get("path", ""))).absolute()
        != synthetic_gate_path
        or synthetic_record.get("sha256") != sha256_file(synthetic_gate_path)
        or streaming_contract.get("protocol_id") != A4_PROTOCOL_ID
        or streaming_contract.get("schema_sha256") != streaming_schema_sha256
        or streaming_contract.get("physical_chunk_rows") != PHYSICAL_CHUNK_ROWS
    ):
        raise RuntimeError("completed streaming protocol/gate provenance changed")
    manifest_streaming = completed.get("streaming", {})
    if (
        manifest_streaming.get("protocol_id") != A4_PROTOCOL_ID
        or manifest_streaming.get("schema_sha256") != streaming_schema_sha256
        or manifest_streaming.get("physical_chunk_rows") != PHYSICAL_CHUNK_ROWS
        or manifest_streaming.get("synthetic_gate_sha256")
        != sha256_file(synthetic_gate_path)
        or manifest_streaming.get("whole_table_materialization_detected") is not False
    ):
        raise RuntimeError("completed A5 streaming manifest binding changed")

    if (
        locked_payload.get("schema_version") != SCHEMA_VERSION
        or locked_payload.get("protocol_id") != PROTOCOL_ID
        or locked_payload.get("status")
        != "A5_INPUT_LOCK_GO_NO_MODEL_OUTPUT"
        or locked_payload.get("execution_identity") != dict(execution_identity)
        or expanded_row_absence_proof(locked_payload)
        != locked_payload.get("expanded_row_absence_proof")
        or sidecar_record.get("semantic_sha256") != nested_sha256(locked_payload)
        or sidecar_record.get("contract_version") != SCHEMA_VERSION
        or sidecar_record.get("item_count") != len(locked_payload.get("items", []))
        or sidecar_record.get("expanded_logical_rows_present") is not False
    ):
        raise RuntimeError("completed compact sidecar semantic binding changed")
    census_manifest, census_records = _verify_a5_census_artifacts(
        repo_root=repo_root,
        output_root=output_root,
        completed=completed,
        locked_payload=locked_payload,
        execution_identity=execution_identity,
    )

    recorded_runtime_assets = locked_payload.get("runtime_assets", {})
    current_runtime_assets: dict[str, Any] = {}
    for role in ("receiver", "sender"):
        recorded = recorded_runtime_assets.get(role)
        if not isinstance(recorded, Mapping):
            raise RuntimeError("completed runtime-asset provenance is incomplete")
        current = tokenizer_runtime_asset_tree(
            Path(str(recorded.get("requested_path", "")))
        )
        current_runtime_assets[role] = {
            "model_id": recorded.get("model_id"),
            **current,
        }
        if current_runtime_assets[role] != dict(recorded):
            raise RuntimeError(f"completed {role} runtime asset tree changed")
        tokenizer_record = locked_payload.get("tokenizers", {}).get(role, {})
        if (
            tokenizer_record.get("name") != recorded.get("model_id")
            or Path(str(tokenizer_record.get("path", ""))).absolute()
            != Path(current["requested_path"]).absolute()
            or tokenizer_record.get("files")
            != _tokenizer_files(Path(current["requested_path"]))
        ):
            raise RuntimeError(f"completed {role} tokenizer provenance changed")

    current_assets = input_asset_state(
        repo_root=repo_root,
        e0_data_root=e0_data_root,
        runtime_assets=current_runtime_assets,
        source_snapshot_verification=source_receipt["verification"],
    )
    if locked_payload.get("input_asset_state") != current_assets:
        raise RuntimeError("completed input-asset/provenance binding changed")
    provenance_record = completed.get("provenance", {})
    if (
        provenance_record.get("input_assets_before_sha256")
        != current_assets["aggregate_sha256"]
        or provenance_record.get("input_assets_after_sha256")
        != current_assets["aggregate_sha256"]
        or provenance_record.get("input_assets_unchanged") is not True
        or provenance_record.get("materialized_e0_dev_data_tree_sha256")
        != current_assets["e0_data_assets"]["tree_sha256"]
        or provenance_record.get("runtime_prompt_assets_sha256")
        != nested_sha256(
            locked_payload.get("a5_prompt_provenance", {}).get(
                "runtime_prompt_assets"
            )
        )
        or provenance_record.get("prompt_config_identity_sha256")
        != nested_sha256(
            locked_payload.get("a5_prompt_provenance", {}).get(
                "prompt_config_identity"
            )
        )
        or provenance_record.get("renderer_source_identity_sha256")
        != nested_sha256(
            locked_payload.get("a5_prompt_provenance", {}).get(
                "renderer_identity"
            )
        )
    ):
        raise RuntimeError("completed A5 prompt provenance binding changed")
    source_record = locked_payload.get("source", {})
    if (
        Path(str(source_record.get("e0_data_root", ""))).absolute()
        != e0_data_root.absolute()
    ):
        raise RuntimeError("completed E0-design data root changed")
    a5_contract = _load_a5_prompt_contract(repo_root)
    prompt_config_identity = _verify_all_e0_prompt_configs(repo_root, a5_contract)
    locked_prompt_provenance = locked_payload.get("a5_prompt_provenance", {})
    if locked_prompt_provenance.get("prompt_config_identity") != prompt_config_identity:
        raise RuntimeError("completed A5 prompt-config identity changed")
    current_renderer_source = attest_e0_renderer_identity(repo_root)
    locked_renderer = locked_prompt_provenance.get("renderer_identity", {})
    for name, value in current_renderer_source.items():
        if name in {
            "production_renderer_exactly_attested",
            "production_renderer_exact_attestation_pending",
        }:
            continue
        if locked_renderer.get(name) != value:
            raise RuntimeError("completed A5 frozen renderer source identity changed")
    if (
        locked_renderer.get("historical_oracle_mode")
        != "EXACT_FROZEN_SOURCE_IDENTITY"
        or locked_renderer.get("renderer_source_identity_attested") is not True
        or locked_renderer.get("production_renderer_exactly_attested") is not True
        or locked_renderer.get("full_row_replay_group_count")
        != sum(TASK_GROUP_COUNTS.values())
        or locked_renderer.get("prechat_prompt_replay_equal") is not True
        or locked_renderer.get("receiver_sender_rendered_replay_equal") is not True
        or locked_renderer.get("production_alignment_replay_equal") is not True
    ):
        raise RuntimeError("completed A5 renderer replay attestation is incomplete")
    split = load_e0_design_lock(
        repo_root / "recipe/eval_recipe/fpct_e1/e1_data_split_manifest.json"
    )
    dev = verify_e0_dev_anchor(repo_root / E0_DEV_MANIFEST_RELATIVE, split)
    if (
        source_record.get("split_sha256") != split["sha256"]
        or source_record.get("dev_manifest_sha256") != dev["sha256"]
    ):
        raise RuntimeError("completed split/dev provenance binding changed")

    geometry_lock = streaming_contract.get("geometry_lock", {})
    geometry_samples = output_root / "input_geometry_samples.parquet"
    geometry_manifest_path = output_root / "input_geometry_manifest.json"
    geometry_receipt_path = output_root / "input_geometry_receipt.json"
    _verify_completed_file_record(
        geometry_lock.get("samples", {}), geometry_samples, "geometry Parquet"
    )
    _verify_completed_file_record(
        geometry_lock.get("manifest", {}), geometry_manifest_path, "geometry manifest"
    )
    _verify_completed_file_record(
        geometry_lock.get("receipt", {}), geometry_receipt_path, "geometry receipt"
    )
    geometry_manifest = json.loads(geometry_manifest_path.read_text(encoding="utf-8"))
    geometry_receipt = json.loads(geometry_receipt_path.read_text(encoding="utf-8"))
    validate_streaming_schema_artifact(geometry_manifest, streaming_schema_path)
    validate_streaming_schema_artifact(geometry_receipt, streaming_schema_path)
    if (
        geometry_manifest.get("execution_identity_sha256") != identity_sha256
        or geometry_manifest.get("input_asset_state_sha256")
        != current_assets["aggregate_sha256"]
        or geometry_manifest.get("schema_sha256") != streaming_schema_sha256
        or geometry_receipt.get("status") != "GO"
        or geometry_receipt.get("manifest_sha256")
        != sha256_file(geometry_manifest_path)
        or geometry_receipt.get("execution_identity_sha256") != identity_sha256
        or geometry_receipt.get("input_asset_state_sha256")
        != current_assets["aggregate_sha256"]
        or geometry_receipt.get("schema_sha256") != streaming_schema_sha256
    ):
        raise RuntimeError("completed geometry manifest/receipt binding changed")

    sidecar_items = locked_payload.get("items")
    if not isinstance(sidecar_items, Sequence) or isinstance(sidecar_items, (str, bytes)):
        raise RuntimeError("completed compact sidecar items are missing")
    expected_geometry = {
        str(item["sample_sha256"]): _geometry_row(_to_python(item))
        for item in sidecar_items
    }
    if len(expected_geometry) != len(sidecar_items):
        raise RuntimeError("completed compact sidecar contains duplicate samples")
    observed_geometry: dict[str, dict[str, Any]] = {}
    ordered_geometry: list[dict[str, Any]] = []
    for row in _verified_geometry_rows(
        samples_path=geometry_samples,
        manifest_path=geometry_manifest_path,
        schema_sha256=streaming_schema_sha256,
    ):
        sample_sha = str(row["sample_sha256"])
        if sample_sha in observed_geometry:
            raise RuntimeError("completed geometry contains duplicate samples")
        observed_geometry[sample_sha] = row
        ordered_geometry.append(row)
    if observed_geometry != expected_geometry:
        raise RuntimeError("completed geometry differs from compact sidecar")
    for item in sidecar_items:
        semantic_source = {
            key: value
            for key, value in _to_python(item).items()
            if key not in {"feature", "item_semantic_sha256"}
        }
        if (
            item.get("item_semantic_sha256") != nested_sha256(semantic_source)
            or item.get("answer_query_sequence_sha256")
            != nested_sha256(item.get("answer_queries"))
            or item.get("parent_sequence_sha256")
            != nested_sha256(item.get("certified_parents"))
            or item.get("raw_topology_compact_sha256")
            != nested_sha256(item.get("raw_topology_compact"))
        ):
            raise RuntimeError("completed item semantic/provenance hash changed")
    expected_sample_sequence_sha256 = nested_sha256(
        [
            [
                row["sample_sha256"],
                row["content_group_sha256"],
                row["item_semantic_sha256"],
                row["raw_topology_compact_sha256"],
            ]
            for row in ordered_geometry
        ]
    )
    expected_row_volume = _global_row_volume(sidecar_items)
    expected_physical_rows = sum(int(row["N_s"]) for row in ordered_geometry)
    expected_disk_bytes = (
        expected_physical_rows
        * int(synthetic_gate["estimated_physical_bytes_per_row"])
    )
    expected_inode_count = (
        sum(int(row["expected_chunk_count"]) for row in ordered_geometry)
        + 3 * len(ordered_geometry)
        + 12
    )
    observed_task_counts = {
        task: sum(row["task"] == task for row in ordered_geometry) for task in TASKS
    }
    if (
        geometry_manifest.get("population") != "e0_design"
        or geometry_manifest.get("task_group_counts") != observed_task_counts
        or observed_task_counts != TASK_GROUP_COUNTS
        or geometry_manifest.get("sample_count") != len(ordered_geometry)
        or geometry_manifest.get("sample_sequence_sha256")
        != expected_sample_sequence_sha256
        or geometry_manifest.get("row_volume") != expected_row_volume
        or geometry_manifest.get("estimated_physical_rows")
        != expected_physical_rows
        or geometry_manifest.get("estimated_disk_bytes") != expected_disk_bytes
        or geometry_manifest.get("estimated_inode_count") != expected_inode_count
        or geometry_manifest.get("disk_preflight_passed") is not True
        or geometry_manifest.get("inode_preflight_passed") is not True
        or geometry_lock.get("row_volume") != expected_row_volume
        or geometry_lock.get("estimated_disk_bytes") != expected_disk_bytes
        or geometry_lock.get("estimated_inode_count") != expected_inode_count
    ):
        raise RuntimeError("completed geometry metadata/provenance changed")
    expected_geometry_receipt = {
        "schema_version": 6,
        "protocol_id": A4_PROTOCOL_ID,
        "artifact_type": "input_geometry_receipt",
        "status": "GO",
        "manifest_sha256": sha256_file(geometry_manifest_path),
        "schema_sha256": streaming_schema_sha256,
        "sample_count": len(ordered_geometry),
        "expanded_logical_rows_materialized": False,
        "task_counts_exact_128_70_128": True,
        "input_assets_unchanged_during_lock": True,
        "input_asset_state_before_sha256": current_assets["aggregate_sha256"],
        "input_asset_state_after_sha256": current_assets["aggregate_sha256"],
        "input_asset_state_sha256": current_assets["aggregate_sha256"],
        "execution_identity_sha256": identity_sha256,
        "bounded_geometry_replay_equal": True,
        "pass_b_geometry_source_only": "verified_input_geometry_samples.parquet",
        "e1_pilot_consumed": False,
        "model_or_checkpoint_loaded": False,
        "gpu_or_kubernetes_used": False,
    }
    if geometry_receipt != expected_geometry_receipt:
        raise RuntimeError("completed geometry receipt semantics changed")

    sidecar_firewall_false = (
        "e1_pilot_consumed",
        "e1_pilot_rendered_tokenized_aligned_run_or_read",
        "model_selection_consumed",
        "test_consumed",
        "confirmatory_consumed",
        "model_or_checkpoint_loaded",
        "gpu_or_cuda_used",
        "training",
    )
    if any(
        locked_payload.get("firewall", {}).get(name) is not False
        for name in sidecar_firewall_false
    ):
        raise RuntimeError("completed sidecar firewall provenance changed")
    manifest_firewall_false = (
        "old_execution_artifact_reused",
        "model_instantiated",
        "model_or_checkpoint_loaded",
        "model_forward_run",
        "gpu_or_cuda_used",
        "kubernetes_used",
        "training",
        "e1_pilot_consumed",
        "confirmatory_consumed",
    )
    if any(
        completed.get("firewall", {}).get(name) is not False
        for name in manifest_firewall_false
    ):
        raise RuntimeError("completed A5 manifest firewall provenance changed")
    if (
        locked_payload.get("e1_pilot_consumed") is not False
        or locked_payload.get("model_or_checkpoint_loaded") is not False
        or locked_payload.get("cuda_initialized") is not False
    ):
        raise RuntimeError("completed sidecar execution firewall changed")

    dimensions = locked_payload.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise RuntimeError("completed receiver dimensions are missing")
    verified_template_lock = _write_streaming_template_lock(
        output_dir=output_root,
        dimensions=dimensions,
        schema_sha256=streaming_schema_sha256,
        geometry_lock=geometry_lock,
        execution_identity=execution_identity,
        input_assets_before_sha256=current_assets["aggregate_sha256"],
        input_assets_after_sha256=current_assets["aggregate_sha256"],
        synthetic_gate_path=synthetic_gate_path,
        synthetic_gate=synthetic_gate,
        verify_only=True,
    )
    if (
        verified_template_lock != streaming_contract.get("streaming_template_lock")
        or locked_payload.get("streaming_contract", {}).get("geometry_lock")
        != geometry_lock
        or locked_payload.get("streaming_contract", {}).get(
            "streaming_template_lock"
        )
        != verified_template_lock
    ):
        raise RuntimeError("completed streaming template binding changed")
    if (
        completed.get("streaming", {}).get("geometry_lock_receipt_sha256")
        != sha256_file(geometry_receipt_path)
        or completed.get("streaming", {}).get(
            "streaming_template_lock_receipt_sha256"
        )
        != verified_template_lock["receipt"]["sha256"]
        or completed.get("streaming", {}).get("logical_row_coverage_exact")
        is not True
        or completed.get("streaming", {}).get("streaming_semantic_replay_equal")
        is not True
    ):
        raise RuntimeError("completed A5 streaming receipt SHA binding changed")

    expected_root_entries = {
        RUN_IDENTITY_NAME,
        geometry_samples.name,
        geometry_manifest_path.name,
        geometry_receipt_path.name,
        "row_templates",
        "input_row_template_chunk_index.json",
        "streaming_input_lock_receipt.json",
        PROMPT_CENSUS_RECORDS_NAME,
        PROMPT_CENSUS_NAME,
        output_sidecar.name,
        output_manifest.name,
    }
    go_receipt_path = output_root / GO_RECEIPT_NAME
    if expect_go_receipt:
        expected_root_entries.add(GO_RECEIPT_NAME)
        if not go_receipt_path.is_file() or go_receipt_path.is_symlink():
            raise RuntimeError("completed A5 input lock lacks GO receipt")
        go_receipt = json.loads(go_receipt_path.read_text(encoding="utf-8"))
        validate_a5_schema_artifact(go_receipt, repo_root=repo_root)
        if (
            go_receipt.get("execution_sha") != execution_identity["execution_sha"]
            or go_receipt.get("run_uid") != execution_identity["run_uid"]
            or go_receipt.get("run_root") != execution_identity["run_root"]
            or go_receipt.get("prompt_census_manifest_sha256")
            != sha256_file(output_root / PROMPT_CENSUS_NAME)
            or go_receipt.get("checks") != completed.get("hard_gate_checks")
            or go_receipt.get("zero_counts") != completed.get("zero_counts")
            or go_receipt.get("downstream_e1_2_authorized") is not False
        ):
            raise RuntimeError("completed A5 GO receipt binding changed")
    elif go_receipt_path.exists():
        raise RuntimeError("pre-GO independent verifier found a premature GO receipt")
    observed_root_entries = {path.name for path in output_root.iterdir()}
    if observed_root_entries != expected_root_entries:
        raise RuntimeError("completed input root contains missing or unbound artifacts")
    return completed


def _prepare_input_lock_after_identity(
    *,
    repo_root: Path,
    e0_data_root: Path,
    output_sidecar: Path,
    output_manifest: Path,
    execution_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize the local sidecar; caller must run only after code lock."""

    if output_sidecar.absolute().parent != output_manifest.absolute().parent:
        raise ValueError("A5 input-lock sidecar and manifest must share one root")
    output_root = _canonical_real_directory(
        output_manifest.absolute().parent, "A5 input-lock producer root"
    )
    producer_finals = (
        output_sidecar.absolute(),
        output_manifest.absolute(),
        output_root / "input_geometry_samples.parquet",
        output_root / "input_geometry_manifest.json",
        output_root / "input_geometry_receipt.json",
        output_root / "input_row_template_chunk_index.json",
        output_root / "streaming_input_lock_receipt.json",
        output_root / PROMPT_CENSUS_RECORDS_NAME,
        output_root / PROMPT_CENSUS_NAME,
        output_root / GO_RECEIPT_NAME,
    )
    for path in producer_finals:
        _preflight_producer_file(path, "A5 input-lock producer final")
    row_template_root = output_root / "row_templates"
    try:
        row_template_root.lstat()
    except FileNotFoundError:
        pass
    else:
        _canonical_real_directory(
            row_template_root, "A4 row-template producer root"
        )
    if output_manifest.exists():
        return _verify_completed_input_lock(
            repo_root=repo_root,
            e0_data_root=e0_data_root,
            output_sidecar=output_sidecar,
            output_manifest=output_manifest,
            execution_identity=execution_identity,
        )
    import torch
    from transformers import AutoTokenizer
    from rosetta.train.dataset_adapters import AlignedChatDataset
    from rosetta.utils.evaluate import set_default_chat_template
    from rosetta.utils.model_loading import resolve_model_path
    from script.evaluation.unified_evaluator import UnifiedEvaluator

    if torch.cuda.is_initialized():
        raise RuntimeError("prepare-input-lock refuses an initialized CUDA runtime")
    streaming_schema_path = repo_root / A4_STREAMING_SCHEMA_RELATIVE
    synthetic_gate_path = repo_root / A5_SYNTHETIC_GATE_RELATIVE
    if not streaming_schema_path.is_file() or not synthetic_gate_path.is_file():
        raise FileNotFoundError("A5 streaming schema/synthetic gate is unavailable")
    streaming_schema_sha256 = sha256_file(streaming_schema_path)
    from script.analysis.fpct_e1_a5_prompt_gate import verify_a5_gate

    synthetic_gate = verify_a5_gate(synthetic_gate_path, repo_root=repo_root)
    if (
        synthetic_gate.get("protocol_id") != A5_PROTOCOL_ID
        or synthetic_gate.get("status") != "GO_PRE_NATURAL_SYNTHETIC_HARD_GATE"
        or synthetic_gate.get("natural_e0_design_accessed") is not False
        or synthetic_gate.get("model_or_checkpoint_loaded") is not False
        or synthetic_gate.get("gpu_or_kubernetes_used") is not False
    ):
        raise ValueError("A5 pre-natural synthetic gate is absent or incompatible")
    renderer_identity = attest_e0_renderer_identity(repo_root)
    a5_contract = _load_a5_prompt_contract(repo_root)
    prompt_config_identity = _verify_all_e0_prompt_configs(repo_root, a5_contract)
    split = load_e0_design_lock(
        repo_root / "recipe/eval_recipe/fpct_e1/e1_data_split_manifest.json"
    )
    dev = verify_e0_dev_anchor(repo_root / E0_DEV_MANIFEST_RELATIVE, split)
    receiver_name = "Qwen/Qwen3-0.6B"
    sender_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    receiver_path = Path(resolve_model_path(receiver_name))
    sender_path = Path(resolve_model_path(sender_name))
    runtime_assets = {
        "receiver": {
            "model_id": receiver_name,
            **tokenizer_runtime_asset_tree(receiver_path),
        },
        "sender": {
            "model_id": sender_name,
            **tokenizer_runtime_asset_tree(sender_path),
        },
    }
    source_verification = execution_identity["source_snapshot_receipt"][
        "verification"
    ]
    input_assets_before = input_asset_state(
        repo_root=repo_root,
        e0_data_root=e0_data_root,
        runtime_assets=runtime_assets,
        source_snapshot_verification=source_verification,
    )
    receiver = AutoTokenizer.from_pretrained(receiver_path)
    sender = AutoTokenizer.from_pretrained(sender_path)
    chat_templates_before = {
        "receiver": getattr(receiver, "chat_template", None),
        "sender": getattr(sender, "chat_template", None),
    }
    set_default_chat_template(receiver, receiver_name)
    set_default_chat_template(sender, sender_name)
    chat_templates_after = {
        "receiver": getattr(receiver, "chat_template", None),
        "sender": getattr(sender, "chat_template", None),
    }
    if chat_templates_before != chat_templates_after or any(
        not isinstance(value, str) or not value
        for value in chat_templates_before.values()
    ):
        raise ValueError("A5 runtime used or changed a chat-template fallback")
    runtime_prompt_assets = _attest_a5_runtime_assets(
        contract=a5_contract,
        runtime_assets=runtime_assets,
        receiver=receiver,
        sender=sender,
        e0_data_tree_sha256=input_assets_before["e0_data_assets"]["tree_sha256"],
    )
    runtime_prompt_assets["chat_template_fallback_used"] = False
    dimensions = _config_dimensions(receiver_path)

    reference_config = yaml.safe_load(
        (repo_root / "recipe/eval_recipe/fpct_e0/rendered/eval_2026072201_Y_FF_ai2-arc.yaml").read_text(
            encoding="utf-8"
        )
    )
    rosetta_config = reference_config["model"]["rosetta_config"]
    aligner = _build_aligner(receiver, sender, rosetta_config)
    dataset_cache: dict[tuple[str, str], Any] = {}
    items: list[dict[str, Any]] = []
    task_counts = {task: 0 for task in TASKS}
    ordered_descriptors = [
        (group, descriptor)
        for task_name in TASKS
        for group, descriptor in sorted(dev["records"].items())
        if descriptor["task"] == task_name
    ]
    if len(ordered_descriptors) != sum(TASK_GROUP_COUNTS.values()):
        raise ValueError("A5 census population/order is incomplete")
    census_records: list[dict[str, Any]] = []
    for group_ordinal, (group, descriptor) in enumerate(
        ordered_descriptors, start=1
    ):
        if group_ordinal < 1:
            raise AssertionError("A5 input lock did not restart from group 1")
        task = descriptor["task"]
        example = _load_task_example(
            task=task,
            descriptor=descriptor,
            data_root=e0_data_root,
            cache=dataset_cache,
        )
        question, choices = _question_choices(task, example)
        production_question, production_choices, production_labels = (
            full_question_choices(task, example)
        )
        if question != production_question or choices != production_choices[:4]:
            raise ValueError("A5 historical first-four projection changed")
        historical_content_sha256 = canonical_content_sha256(question, choices)
        if historical_content_sha256 != group:
            raise ValueError("input-lock materialized content hash mismatch")
        expected_sample = canonical_sample_sha256(
            task, descriptor["subject"], descriptor["source_row_id"]
        )
        if expected_sample != descriptor["sample_sha256"]:
            raise ValueError("input-lock sample hash mismatch")
        formatter = UnifiedEvaluator.__new__(UnifiedEvaluator)
        formatter.dataset_name = task
        formatter.model_config = reference_config["model"]
        formatter.eval_config = {
            **reference_config["eval"],
            "dataset": task,
            "use_cot": False,
            "use_template": True,
        }
        projected_example = historical_projected_example(task, example)
        historical_prompt = formatter.format_example(
            projected_example, use_cot=False
        )
        historical_prompt_details = _prompt_alignment_details(
            aligner, historical_prompt, top_k=4
        )
        historical_rendered_sha = _sha256_bytes(
            historical_prompt_details["slm_text"].encode("utf-8")
        )
        historical_alignment_sha = _alignment_sha256(historical_prompt_details)
        if historical_rendered_sha != descriptor["rendered_prompt_sha256"]:
            raise ValueError("A5 historical first-four rendered prompt SHA mismatch")
        if historical_alignment_sha != descriptor["prompt_alignment_sha256"]:
            raise ValueError("A5 historical first-four alignment SHA mismatch")

        # Exact frozen source/config/tokenizer identity establishes that the
        # active renderer is the effective historical E0 renderer; this is not
        # a second independent implementation.  Two calls below are a strict
        # determinism replay for each full row and both tokenizer renderings.
        historical_e0_full_prompt = formatter.format_example(example, use_cot=False)
        a5_production_prompt = formatter.format_example(example, use_cot=False)
        if historical_e0_full_prompt.encode("utf-8") != a5_production_prompt.encode(
            "utf-8"
        ):
            raise ValueError("A5 full pre-chat prompt differs from historical E0")
        historical_e0_full_details = _prompt_alignment_details(
            aligner, historical_e0_full_prompt, top_k=4
        )
        prompt = a5_production_prompt
        prompt_details = _prompt_alignment_details(aligner, prompt, top_k=4)
        if _to_python(historical_e0_full_details) != _to_python(prompt_details):
            raise ValueError(
                "A5 receiver/sender rendered bytes or production alignment replay changed"
            )
        if (
            historical_e0_full_details.get("slm_text")
            != prompt_details.get("slm_text")
            or historical_e0_full_details.get("llm_text")
            != prompt_details.get("llm_text")
        ):
            raise ValueError("A5 dual-tokenizer rendered chat bytes changed")
        rendered_sha = _sha256_bytes(prompt_details["slm_text"].encode("utf-8"))
        production_alignment_sha = _alignment_sha256(prompt_details)
        answer = formatter.parse_answer(example)
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError("input-lock has no canonical A-D gold answer")
        gold_response = GOLD_RESPONSE_TEMPLATE.format(answer=answer)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": gold_response},
        ]
        raw_full_details = aligner.align_chat_messages_soft(
            messages,
            add_generation_prompt=False,
            return_details=True,
            apply_confidence_control=False,
            top_k=4,
        )
        full_details = aligner.sanitize_fpct_soft_alignment(
            raw_full_details,
            target_length=len(raw_full_details["slm_ids"]),
            source_length=len(raw_full_details["llm_ids"]),
        )
        aligned = AlignedChatDataset(
            [messages],
            aligner,
            max_length=32768,
            soft_alignment_top_k=4,
            fpct_alignment_sanitizer="certified_slot0_v1",
        )
        feature = aligned[0]
        _verify_feature_alignment(feature, full_details)
        instruction_end = _instruction_end(feature["labels"])
        provenance = feature_provenance(feature, full_details, gold_response)
        full_soft = full_details["soft_alignment"]
        alignment_lock = {
            "source_indices": full_soft["source_indices"],
            "source_weights": full_soft["source_weights"],
            "fpct_certified_mask": full_soft["fpct_certified_mask"],
            "fpct_offset_uncertified_mask": full_soft["fpct_offset_uncertified_mask"],
            "fpct_certification_reason": full_soft["fpct_certification_reason"],
        }
        topology = topology_contract(full_details, instruction_end)
        raw_topology = raw_topology_ledger(
            raw_details=raw_full_details,
            sanitized_details=full_details,
            instruction_end=instruction_end,
            task=task,
            sample_sha256=expected_sample,
            content_group_sha256=group,
            candidate_window=int(rosetta_config.get("soft_alignment_candidate_window", 0)),
        )
        formatter.model_config = reference_config["model"]
        prompt_prepared = formatter.prepare_model_inputs(
            prompt, receiver, torch.device("cpu"), "rosetta", sender, "generate"
        )
        prompt_inputs = prompt_prepared["inputs"]
        answer_queries = answer_query_contract(feature["labels"])
        certified_parents = certified_parent_contract(
            topology,
            source_indices=full_soft["source_indices"],
            source_weights=full_soft["source_weights"],
        )
        expected_rows = expected_long_form_rows(
            answer_query_count=len(answer_queries),
            certified_parent_count=len(certified_parents),
            num_layers=dimensions["num_hidden_layers"],
            num_query_heads=dimensions["num_attention_heads"],
        )
        raw_topology_compact = _raw_topology_compact_metadata(raw_topology)
        census_record = dual_anchor_record(
            task=task,
            content_group_sha256=group,
            sample_key_sha256=expected_sample,
            source_row_id=str(descriptor["source_row_id"]),
            example=example,
            gold_answer=answer,
            historical_rendered_prompt=historical_prompt_details["slm_text"],
            historical_alignment_sha256=historical_alignment_sha,
            production_rendered_prompt=prompt_details["slm_text"],
            production_alignment_sha256=production_alignment_sha,
            historical_prompt_token_count=len(historical_prompt_details["slm_ids"]),
            production_prompt_token_count=len(prompt_details["slm_ids"]),
            production_certified_parent_count=len(certified_parents),
            production_logical_row_count=expected_rows,
            production_physical_chunk_count=math.ceil(
                expected_rows / PHYSICAL_CHUNK_ROWS
            ),
            historical_content_sha256=historical_content_sha256,
        )
        if census_record["raw_choice_labels"] != production_labels:
            raise ValueError("A5 census lost production choice labels/order")
        census_records.append(census_record)
        item = {
            "task": task,
            "sample_sha256": expected_sample,
            "content_group_sha256": group,
            "descriptor": descriptor,
            "gold_answer": answer,
            "gold_response": gold_response,
            "feature": feature,
            "alignment_lock": alignment_lock,
            "prompt_generation_inputs": prompt_inputs,
            "prompt_generation_inputs_sha256": nested_sha256(prompt_inputs),
            "provenance": provenance,
            "answer_queries": answer_queries,
            "certified_parents": certified_parents,
            "Q_s": len(answer_queries),
            "P_s": len(certified_parents),
            "N_s": expected_rows,
            "answer_query_sequence_sha256": nested_sha256(answer_queries),
            "parent_sequence_sha256": nested_sha256(certified_parents),
            "expected_chunk_count": math.ceil(expected_rows / PHYSICAL_CHUNK_ROWS),
            "expected_long_form_rows": expected_rows,
            "raw_topology_ledger": raw_topology,
            "raw_topology_compact": raw_topology_compact,
            "raw_topology_compact_sha256": nested_sha256(raw_topology_compact),
            "instruction_end": instruction_end,
            "rendered_prompt_sha256": rendered_sha,
            "production_rendered_prompt_sha256": rendered_sha,
            "historical_rendered_prompt_sha256": historical_rendered_sha,
            "production_alignment_sha256": production_alignment_sha,
            "historical_alignment_sha256": historical_alignment_sha,
            "raw_full_row_sha256": raw_full_row_sha256(example),
            "prompt_relation": census_record["prompt_relation"],
            "choice_difference_only": census_record["choice_difference_only"],
        }
        item["item_semantic_sha256"] = nested_sha256(
            {key: value for key, value in item.items() if key != "feature"}
        )
        items.append(item)
        task_counts[task] += 1
    if task_counts != TASK_GROUP_COUNTS:
        raise ValueError("input-lock population differs from frozen 128/70/128")
    from script.analysis.fpct_e1_a5_prompt_gate import validate_a5_schema_artifact

    for census_record in census_records:
        validate_a5_schema_artifact(census_record, repo_root=repo_root)
    census_records_path = output_root / PROMPT_CENSUS_RECORDS_NAME
    census_records_payload = b"".join(
        canonical_json_bytes(record) for record in census_records
    )
    census_records_sha256 = publish_bytes_no_overwrite(
        census_records_path, census_records_payload
    )
    census_manifest = build_census_manifest(
        census_records,
        execution_sha=str(execution_identity["execution_sha"]),
        run_uid=str(execution_identity["run_uid"]),
        record_artifact={
            "relative_path": PROMPT_CENSUS_RECORDS_NAME,
            "sha256": census_records_sha256,
            "bytes": len(census_records_payload),
            "row_count": len(census_records),
        },
        expected_task_counts=TASK_GROUP_COUNTS,
    )
    validate_a5_schema_artifact(census_manifest, repo_root=repo_root)
    census_manifest_path = output_root / PROMPT_CENSUS_NAME
    atomic_json(census_manifest_path, census_manifest)
    for role, model_path in (("receiver", receiver_path), ("sender", sender_path)):
        if portable_runtime_asset_tree(
            tokenizer_runtime_asset_tree(model_path)
        ) != portable_runtime_asset_tree(runtime_assets[role]):
            raise ValueError(f"{role} runtime asset tree changed during CPU input locking")
    from script.experiment.fpct_e1_source_snapshot_lock import (
        verify_source_snapshot_receipt,
    )

    source_verification_after = verify_source_snapshot_receipt(
        Path(execution_identity["source_snapshot_receipt"]["path"]),
        Path(execution_identity["source_snapshot_root"]),
        str(execution_identity["execution_sha"]),
    )
    input_assets_after = input_asset_state(
        repo_root=repo_root,
        e0_data_root=e0_data_root,
        runtime_assets=runtime_assets,
        source_snapshot_verification=source_verification_after,
    )
    if input_assets_before != input_assets_after:
        raise RuntimeError("input assets changed during CPU input locking")
    task_rank = {task: index for index, task in enumerate(TASKS)}
    items.sort(
        key=lambda item: (
            task_rank[item["task"]],
            item["content_group_sha256"],
            item["sample_sha256"],
        )
    )
    by_task = {}
    for task in TASKS:
        members = [item for item in items if item["task"] == task]
        row_volume = expected_long_form_row_volume(members)
        taxonomy_counts: dict[str, int] = {}
        raw_count = 0
        runtime_m_ge2 = 0
        offset_uncertified = 0
        raw_m_distribution = {str(value): 0 for value in range(2, 5)}
        runtime_m_distribution = {str(value): 0 for value in range(1, 5)}
        raw_digest = hashlib.sha256()
        for item in members:
            for row in item["raw_topology_ledger"]:
                raw_digest.update(canonical_json_bytes(row))
                raw_count += 1
                runtime_m_ge2 += int(row["runtime_candidate_count"] >= 2)
                offset_uncertified += int(not row["certified"])
                raw_m_distribution[str(row["raw_candidate_count"])] += 1
                runtime_m_distribution[str(row["runtime_candidate_count"])] += 1
                taxonomy_counts[row["taxonomy"]] = (
                    taxonomy_counts.get(row["taxonomy"], 0) + 1
                )
        semantic_members = [
            {
                "sample_sha256": item["sample_sha256"],
                "content_group_sha256": item["content_group_sha256"],
                "item_semantic_sha256": item["item_semantic_sha256"],
            }
            for item in members
        ]
        task_group_contract = [
            {
                "content_group_sha256": group,
                "sample_sha256": [record["sample_sha256"]],
            }
            for group, record in dev["records"].items()
            if record["task"] == task
        ]
        by_task[task] = {
            "group_count": len(members),
            "members": semantic_members,
            "membership_sha256": task_membership_sha256(
                task_group_contract, semantic_members
            ),
            "row_template": streaming_row_template_attestation(
                members,
                task=task,
                num_layers=dimensions["num_hidden_layers"],
                num_query_heads=dimensions["num_attention_heads"],
                num_kv_heads=dimensions["num_key_value_heads"],
                schema_sha256=streaming_schema_sha256,
            ),
            "expected_long_form_rows": row_volume,
            "raw_topology": {
                "row_count": raw_count,
                "canonical_row_stream_sha256": raw_digest.hexdigest(),
                "raw_m_ge2_parent_count": raw_count,
                "runtime_m_ge2_parent_count": runtime_m_ge2,
                "raw_m_distribution": raw_m_distribution,
                "runtime_m_distribution": runtime_m_distribution,
                "offset_uncertified_parent_count": offset_uncertified,
                "taxonomy_counts": taxonomy_counts,
                "candidate_window": 0,
                "competing_overlap_requires_explicit_evidence": True,
                "contains_model_output": False,
            },
        }
    geometry_lock = _write_geometry_lock(
        output_dir=output_manifest.parent,
        items=items,
        schema_sha256=streaming_schema_sha256,
        synthetic_gate=synthetic_gate,
        input_assets_before=input_assets_before,
        input_assets_after=input_assets_after,
        execution_identity=execution_identity,
    )
    streaming_template_lock = _write_streaming_template_lock(
        output_dir=output_manifest.parent,
        dimensions=dimensions,
        schema_sha256=streaming_schema_sha256,
        geometry_lock=geometry_lock,
        execution_identity=execution_identity,
        input_assets_before_sha256=input_assets_before["aggregate_sha256"],
        input_assets_after_sha256=input_assets_after["aggregate_sha256"],
        synthetic_gate_path=synthetic_gate_path,
        synthetic_gate=synthetic_gate,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "A5_INPUT_LOCK_GO_NO_MODEL_OUTPUT",
        "split_role": "e0_design",
        "items": items,
        "dimensions": dimensions,
        "streaming_contract": {
            "protocol_id": A4_PROTOCOL_ID,
            "schema_sha256": streaming_schema_sha256,
            "physical_chunk_rows": PHYSICAL_CHUNK_ROWS,
            "expanded_logical_rows_present": False,
            "compact_geometry_only": True,
            "synthetic_gate": _a5_synthetic_gate_binding(synthetic_gate_path),
            "geometry_lock": geometry_lock,
            "streaming_template_lock": streaming_template_lock,
        },
        "execution_identity": dict(execution_identity),
        "a5_prompt_provenance": {
            "renderer_identity": {
                **renderer_identity,
                "historical_oracle_mode": "EXACT_FROZEN_SOURCE_IDENTITY",
                "production_renderer_exactly_attested": True,
                "production_renderer_exact_attestation_pending": None,
                "full_row_replay_group_count": len(census_records),
                "prechat_prompt_replay_equal": True,
                "receiver_sender_rendered_replay_equal": True,
                "production_alignment_replay_equal": True,
            },
            "prompt_config_identity": prompt_config_identity,
            "runtime_prompt_assets": runtime_prompt_assets,
            "census_manifest": {
                "path": str(census_manifest_path),
                "sha256": sha256_file(census_manifest_path),
                "records_path": str(census_records_path),
                "records_sha256": census_records_sha256,
            },
            "historical_projection_anchor_unchanged": True,
            "production_runtime_anchor_operative": True,
        },
        "input_asset_state": input_assets_after,
        "source": {
            "split_sha256": split["sha256"],
            "dev_manifest_sha256": dev["sha256"],
            "e0_data_root": str(e0_data_root),
        },
        "tokenizers": {
            "receiver": {
                "name": receiver_name,
                "path": str(receiver_path),
                "files": _tokenizer_files(receiver_path),
            },
            "sender": {
                "name": sender_name,
                "path": str(sender_path),
                "files": _tokenizer_files(sender_path),
            },
        },
        "runtime_assets": runtime_assets,
        "task_contract": by_task,
        "e1_pilot_consumed": False,
        "model_or_checkpoint_loaded": False,
        "cuda_initialized": False,
        "firewall": {
            "e1_pilot_consumed": False,
            "e1_pilot_rendered_tokenized_aligned_run_or_read": False,
            "model_selection_consumed": False,
            "test_consumed": False,
            "confirmatory_consumed": False,
            "model_or_checkpoint_loaded": False,
            "gpu_cuda_or_kubernetes_used": False,
            "gpu_or_cuda_used": False,
            "training": False,
        },
    }
    payload["expanded_row_absence_proof"] = expanded_row_absence_proof(payload)
    _canonical_real_directory(output_sidecar.parent, "compact sidecar parent")
    _preflight_producer_file(output_sidecar, "compact sidecar final")
    if output_sidecar.exists():
        existing_payload = torch.load(
            output_sidecar, map_location="cpu", weights_only=False
        )
        if nested_sha256(existing_payload) != nested_sha256(payload):
            raise RuntimeError("existing compact sidecar differs after crash replay")
    else:
        descriptor, temporary_text = tempfile.mkstemp(
            prefix=f".{output_sidecar.name}.",
            suffix=".tmp",
            dir=output_sidecar.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_text)
        try:
            torch.save(payload, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            try:
                os.link(temporary, output_sidecar)
            except FileExistsError:
                if (
                    output_sidecar.stat().st_size != temporary.stat().st_size
                    or sha256_file(output_sidecar) != sha256_file(temporary)
                ):
                    raise RuntimeError("immutable compact sidecar winner bytes differ")
            _fsync_directory(output_sidecar.parent)
        finally:
            temporary.unlink(missing_ok=True)
    locked_payload = torch.load(output_sidecar, map_location="cpu", weights_only=False)
    if expanded_row_absence_proof(locked_payload) != payload[
        "expanded_row_absence_proof"
    ]:
        raise RuntimeError("published compact sidecar failed expanded-row absence proof")
    source_tree_sha256 = execution_identity["source_snapshot_receipt"][
        "verification"
    ].get("mounted_tree_canonical_sha256")
    if not isinstance(source_tree_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", source_tree_sha256
    ):
        raise RuntimeError("A5 source snapshot canonical tree SHA is missing")
    expected_group_count = sum(TASK_GROUP_COUNTS.values())
    expected_group_ids = set(dev["records"])
    observed_group_ids = {
        str(record["content_group_sha256"]) for record in census_records
    }
    renderer_lock = payload["a5_prompt_provenance"]["renderer_identity"]
    streaming_receipt = json.loads(
        Path(streaming_template_lock["receipt"]["path"]).read_text(encoding="utf-8")
    )
    hard_gate_checks = {
        "historical_projection_anchor_unchanged": bool(
            payload["a5_prompt_provenance"][
                "historical_projection_anchor_unchanged"
            ]
            and all(
                record["historical_choice_count"] == 4
                for record in census_records
            )
        ),
        "e0_design_membership_unchanged": bool(
            task_counts == TASK_GROUP_COUNTS
            and observed_group_ids == expected_group_ids
            and len(observed_group_ids) == expected_group_count
        ),
        "renderer_source_identity_attested": bool(
            renderer_lock.get("renderer_source_identity_attested") is True
        ),
        "production_renderer_exactly_attested": bool(
            renderer_lock.get("production_renderer_exactly_attested") is True
            and renderer_lock.get("historical_oracle_mode")
            == "EXACT_FROZEN_SOURCE_IDENTITY"
            and renderer_lock.get("full_row_replay_group_count")
            == expected_group_count
        ),
        "production_data_tree_exactly_attested": bool(
            runtime_prompt_assets.get("production_data_tree_exactly_attested")
            is True
            and runtime_prompt_assets.get(
                "materialized_e0_dev_data_tree_sha256"
            )
            == input_assets_after["e0_data_assets"]["tree_sha256"]
        ),
        "all_326_groups_resolved": bool(
            census_manifest["population_count"] == expected_group_count
            and len(census_records) == expected_group_count
        ),
        "all_first4_choices_equal_historical": bool(
            all(
                record["historical_choice_count"] == 4
                and record["production_choice_count"] >= 4
                for record in census_records
            )
        ),
        "all_gold_answers_in_A_B_C_D": bool(
            all(record["gold_answer"] in {"A", "B", "C", "D"} for record in census_records)
        ),
        "all_prompt_differences_classified": bool(
            census_manifest["unexpected_prompt_difference_count"] == 0
            and all(
                record["prompt_relation"]
                in {
                    "EXACT_HISTORICAL_AND_PRODUCTION_MATCH",
                    "EXTRA_CHOICES_ONLY",
                }
                for record in census_records
            )
        ),
        "historical_to_runtime_mapping_complete": bool(
            len(census_manifest["historical_to_production_anchor_map"])
            == expected_group_count
            and len(observed_group_ids) == expected_group_count
        ),
        "production_prompt_replay_equal": bool(
            renderer_lock.get("prechat_prompt_replay_equal") is True
            and renderer_lock.get("receiver_sender_rendered_replay_equal") is True
        ),
        "production_alignment_replay_equal": bool(
            renderer_lock.get("production_alignment_replay_equal") is True
        ),
        "choice_order_preserved": bool(
            all(
                record["raw_choice_labels"]
                == [
                    chr(65 + index)
                    for index in range(record["production_choice_count"])
                ]
                for record in census_records
            )
        ),
        "raw_full_row_hashes_complete": bool(
            all(
                re.fullmatch(r"[0-9a-f]{64}", record["raw_full_row_sha256"])
                is not None
                for record in census_records
            )
        ),
        "logical_row_coverage_exact": bool(
            streaming_template_lock["expected_logical_rows"]
            == streaming_template_lock["emitted_logical_rows"]
            and streaming_receipt.get("expected_logical_rows_eq_emitted") is True
            and streaming_receipt.get("missing_rows") == 0
            and streaming_receipt.get("duplicate_rows") == 0
        ),
        "streaming_semantic_replay_equal": bool(
            streaming_receipt.get("semantic_stream_replay_equal") is True
            and streaming_receipt.get("chunk_partition_semantic_equivalence")
            is True
            and streaming_receipt.get("aggregate_partition_equivalence") is True
        ),
        "bounded_peak_rss": _a5_bounded_peak_rss_gate(
            streaming_receipt, synthetic_gate
        ),
    }
    failed_hard_gates = sorted(
        name for name, passed in hard_gate_checks.items() if passed is not True
    )
    if failed_hard_gates:
        raise RuntimeError(f"A5 derived hard gates failed: {failed_hard_gates}")
    manifest = {
        "schema_version": 7,
        "protocol_id": A5_PROTOCOL_ID,
        "artifact_type": "a5_input_lock_manifest",
        "status": "A5_INPUT_LOCK_GO_NO_MODEL_OUTPUT",
        "split_role": "e0_design",
        "execution": {
            "execution_sha": str(execution_identity["execution_sha"]),
            "run_uid": str(execution_identity["run_uid"]),
            "run_root": str(execution_identity["run_root"]),
            "source_snapshot_receipt_sha256": str(
                execution_identity["source_snapshot_receipt"]["file_sha256"]
            ),
            "source_snapshot_tree_sha256": source_tree_sha256,
        },
        "task_counts": dict(TASK_GROUP_COUNTS),
        "provenance": {
            "renderer_source_identity_sha256": nested_sha256(
                payload["a5_prompt_provenance"]["renderer_identity"]
            ),
            "prompt_config_identity_sha256": nested_sha256(prompt_config_identity),
            "runtime_prompt_assets_sha256": nested_sha256(runtime_prompt_assets),
            "materialized_e0_dev_data_tree_sha256": runtime_prompt_assets[
                "materialized_e0_dev_data_tree_sha256"
            ],
            "input_assets_before_sha256": input_assets_before["aggregate_sha256"],
            "input_assets_after_sha256": input_assets_after["aggregate_sha256"],
            "input_assets_unchanged": input_assets_before == input_assets_after,
        },
        "census": {
            "manifest": {
                "path": str(census_manifest_path),
                "bytes": census_manifest_path.stat().st_size,
                "sha256": sha256_file(census_manifest_path),
            },
            "records": {
                "path": str(census_records_path),
                "bytes": census_records_path.stat().st_size,
                "sha256": sha256_file(census_records_path),
            },
            "record_count": len(census_records),
            "canonical_semantic_stream_sha256": census_manifest[
                "canonical_semantic_stream_sha256"
            ],
        },
        "sidecar": {
            "contract_version": SCHEMA_VERSION,
            "path": str(output_sidecar),
            "bytes": output_sidecar.stat().st_size,
            "file_sha256": sha256_file(output_sidecar),
            "semantic_sha256": nested_sha256(payload),
            "item_count": len(items),
            "expanded_logical_rows_present": False,
        },
        "streaming": {
            "protocol_id": A4_PROTOCOL_ID,
            "schema_sha256": streaming_schema_sha256,
            "physical_chunk_rows": PHYSICAL_CHUNK_ROWS,
            "synthetic_gate_sha256": sha256_file(synthetic_gate_path),
            "geometry_lock_receipt_sha256": geometry_lock["receipt"]["sha256"],
            "streaming_template_lock_receipt_sha256": streaming_template_lock[
                "receipt"
            ]["sha256"],
            "logical_row_coverage_exact": hard_gate_checks[
                "logical_row_coverage_exact"
            ],
            "streaming_semantic_replay_equal": hard_gate_checks[
                "streaming_semantic_replay_equal"
            ],
            "whole_table_materialization_detected": bool(
                streaming_receipt["whole_table_materialization_detected"]
            ),
        },
        "hard_gate_checks": hard_gate_checks,
        "zero_counts": {
            "unexpected_prompt_difference_count": census_manifest[
                "unexpected_prompt_difference_count"
            ],
            "missing_rows": streaming_receipt["missing_rows"],
            "duplicate_rows": streaming_receipt["duplicate_rows"],
        },
        "firewall": {
            "old_execution_artifact_reused": False,
            "model_instantiated": False,
            "model_or_checkpoint_loaded": False,
            "model_forward_run": False,
            "gpu_or_cuda_used": False,
            "kubernetes_used": False,
            "training": False,
            "e1_pilot_consumed": False,
            "confirmatory_consumed": False,
        },
        "e1_2_or_e1_3_authorized": False,
    }
    from script.analysis.fpct_e1_a5_prompt_gate import validate_a5_schema_artifact

    validate_a5_schema_artifact(manifest, repo_root=repo_root)
    atomic_json(output_manifest, manifest)
    verified_manifest = _verify_completed_input_lock(
        repo_root=repo_root,
        e0_data_root=e0_data_root,
        output_sidecar=output_sidecar,
        output_manifest=output_manifest,
        execution_identity=execution_identity,
        expect_go_receipt=False,
    )
    go_receipt = {
        "schema_version": 7,
        "protocol_id": A5_PROTOCOL_ID,
        "artifact_type": "a5_input_lock_receipt",
        "status": "A5_INPUT_LOCK_GO",
        "execution_sha": str(execution_identity["execution_sha"]),
        "run_uid": str(execution_identity["run_uid"]),
        "run_root": str(execution_identity["run_root"]),
        "source_snapshot_receipt_sha256": str(
            execution_identity["source_snapshot_receipt"]["file_sha256"]
        ),
        "prompt_census_manifest_sha256": sha256_file(census_manifest_path),
        "checks": dict(verified_manifest["hard_gate_checks"]),
        "firewall": {
            "old_execution_artifact_reused": False,
            "whole_table_materialization_detected": False,
            "model_instantiated": False,
            "model_or_checkpoint_loaded": False,
            "model_forward_run": False,
            "gpu_or_kubernetes_used": False,
            "e1_pilot_consumed": False,
            "confirmatory_consumed": False,
        },
        "zero_counts": dict(verified_manifest["zero_counts"]),
        "resume_from_group_one": True,
        "downstream_e1_2_authorized": False,
    }
    validate_a5_schema_artifact(go_receipt, repo_root=repo_root)
    atomic_json(output_root / GO_RECEIPT_NAME, go_receipt)
    return _verify_completed_input_lock(
        repo_root=repo_root,
        e0_data_root=e0_data_root,
        output_sidecar=output_sidecar,
        output_manifest=output_manifest,
        execution_identity=execution_identity,
        expect_go_receipt=True,
    )


def _blocked_check_name(error: Exception) -> str:
    text = str(error).split(":", 1)[0].strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return normalized[:160] or error.__class__.__name__.lower()


def _a5_bounded_peak_rss_gate(
    streaming_receipt: Mapping[str, Any], synthetic_gate: Mapping[str, Any]
) -> bool:
    """Bind the natural lock to the A5 gate's frozen streaming evidence."""

    return bool(
        streaming_receipt.get("bounded_peak_rss") is True
        and synthetic_gate.get("streaming_stress", {}).get("bounded_peak_rss")
        is True
    )


def _a5_synthetic_gate_binding(path: Path) -> dict[str, str]:
    canonical = _canonical_regular_file(path, "A5 synthetic gate")
    return {"path": str(canonical), "sha256": sha256_file(canonical)}


def _publish_blocked_receipt(
    output_root: Path,
    error: Exception,
    execution_identity: Mapping[str, Any],
) -> None:
    """Emit the only terminal receipt permitted after a caught gate failure."""

    payload = {
        "schema_version": 7,
        "protocol_id": A5_PROTOCOL_ID,
        "artifact_type": "a5_input_lock_blocked_receipt",
        "status": "A5_INPUT_LOCK_BLOCKED",
        "execution_sha": str(execution_identity["execution_sha"]),
        "run_uid": str(execution_identity["run_uid"]),
        "run_root": str(execution_identity["run_root"]),
        "failed_check": _blocked_check_name(error),
        "failure_detail_sha256": _sha256_bytes(str(error).encode("utf-8")),
        "resume_allowed": False,
        "artifact_reuse_allowed": False,
        "scientific_result": False,
        "downstream_e1_2_or_e1_3_authorized": False,
    }
    from script.analysis.fpct_e1_a5_prompt_gate import validate_a5_schema_artifact

    validate_a5_schema_artifact(payload, repo_root=Path(execution_identity["source_snapshot_root"]))
    atomic_json(output_root / BLOCKED_RECEIPT_NAME, payload)


def prepare_input_lock(
    *,
    repo_root: Path,
    e0_data_root: Path,
    output_sidecar: Path,
    output_manifest: Path,
    execution_sha: str,
    source_snapshot_root: Path,
    source_snapshot_receipt: Path,
    run_uid: str,
    run_root: Path,
    _test_only_sealed_execution: tuple[object, str, str] | None = None,
) -> dict[str, Any]:
    """Validate the successor identity, then run one fail-closed A5 input lock."""

    output_root = output_manifest.absolute().parent
    if output_sidecar.absolute().parent != output_root:
        raise ValueError("A5 input-lock sidecar and manifest must share one root")
    if repo_root.absolute() != source_snapshot_root.absolute():
        raise ValueError("A5 input lock must execute from its immutable source snapshot")
    sealed_prepare_execution = _require_sealed_prepare_execution(
        repo_root=repo_root,
        source_snapshot_root=source_snapshot_root,
        execution_sha=execution_sha,
        test_sentinel=_test_only_sealed_execution,
    )
    identity = validate_a5_execution_identity(
        execution_sha=execution_sha,
        source_snapshot_root=source_snapshot_root,
        source_snapshot_receipt=source_snapshot_receipt,
        run_uid=run_uid,
        run_root=run_root,
        output_root=output_root,
        output_sidecar_name=output_sidecar.name,
        output_manifest_name=output_manifest.name,
        sealed_prepare_execution=sealed_prepare_execution,
        _test_only_run_parent=(
            run_root.absolute().parent
            if sealed_prepare_execution.get("pytest_verified_test_sentinel") is True
            else None
        ),
    )
    try:
        return _prepare_input_lock_after_identity(
            repo_root=repo_root,
            e0_data_root=e0_data_root,
            output_sidecar=output_sidecar,
            output_manifest=output_manifest,
            execution_identity=identity,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        # Abrupt process death/KeyboardInterrupt is intentionally not caught;
        # immutable chunks plus the execution identity then support exact resume.
        _publish_blocked_receipt(output_root, error, identity)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--e0-data-root", type=Path, required=True)
    parser.add_argument("--output-sidecar", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--source-snapshot-root", type=Path, required=True)
    parser.add_argument("--source-snapshot-receipt", type=Path, required=True)
    parser.add_argument("--run-uid", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = prepare_input_lock(
        repo_root=args.repo_root,
        e0_data_root=args.e0_data_root,
        output_sidecar=args.output_sidecar,
        output_manifest=args.output_manifest,
        execution_sha=args.execution_sha,
        source_snapshot_root=args.source_snapshot_root,
        source_snapshot_receipt=args.source_snapshot_receipt,
        run_uid=args.run_uid,
        run_root=args.run_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
