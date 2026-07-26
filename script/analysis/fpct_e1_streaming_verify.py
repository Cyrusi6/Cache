#!/usr/bin/env python3
"""Bounded, deterministic streaming primitives for the FPCT-E1 A4 contract.

This module is deliberately independent of tokenizers, datasets, models, torch,
and parquet.  It defines the logical ordinal/identity contract and a small
canonical JSONL reference artifact used by synthetic tests and by preflight
verification.  Production parquet writers can use the same ordinal, identity,
range, and semantic-hash functions without sharing a physical byte format.

No function materializes the complete logical row table.  Memory is bounded by
one row plus the (small) chunk manifest, rather than by the Cartesian row count.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


PROTOCOL_ID = "fpct_e1_mechanism_audit_v6_representation_preserving_streaming"
SCHEMA_VERSION = 6
PHYSICAL_CHUNK_ROWS = 4096
LOGICAL_ROW_ID_DOMAIN = b"fpct-e1-a4-logical-row-id-v1"
MANIFEST_NAME = "stream_manifest.json"
PARQUET_MANIFEST_NAME = "chunk_manifest.json"
DEFAULT_STREAMING_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "recipe/eval_recipe/fpct_e1/e1_streaming_schema.json"
)

ROW_KEY_COLUMNS = (
    "sample_sha256",
    "content_group_sha256",
    "input_sha256",
    "alignment_sha256",
    "labels_sha256",
    "gold_response_sha256",
    "layer",
    "query_head",
    "kv_head",
    "query_position",
    "target_position",
    "target_token_id",
    "parent_position",
    "candidate_count",
    "topology",
)
ENDPOINT_ID_COLUMNS = (
    "seed",
    "checkpoint_arm",
    "inference_operator",
    "cell",
    "task",
    "lambda_value",
)

PROTECTED_STREAM_COLUMNS = frozenset(
    (
        *ROW_KEY_COLUMNS,
        "prior",
        "candidate_indices",
        "candidate_valid_mask",
        "candidate_slot_weights",
        "statistical_weight",
        "row_ordinal",
        "logical_row_id",
        "endpoint_id",
        "endpoint_row_id",
        "schema_sha256",
    )
)

CRASH_STAGES = (
    "chunk_temporary_half_written",
    "before_chunk_close",
    "before_artifact_rename",
    "before_manifest_write",
    "before_manifest_rename",
)


def _canonical_artifact_directory(path: Path, label: str) -> Path:
    absolute = path.absolute()
    try:
        mode = absolute.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing") from exc
    if (
        not stat.S_ISDIR(mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
    ):
        raise ValueError(f"{label} must be a canonical non-symlink directory")
    return absolute


def _preflight_artifact_file(path: Path, label: str) -> None:
    """Reject aliases/special files without following them; absence is valid."""

    absolute = path.absolute()
    _canonical_artifact_directory(absolute.parent, f"{label} parent")
    try:
        mode = absolute.lstat().st_mode
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
    ):
        raise ValueError(f"{label} must be absent or a canonical regular file")


def _prepare_artifact_root(path: Path) -> Path:
    """Preflight a root and its parent before the sole permitted mkdir."""

    absolute = path.absolute()
    _canonical_artifact_directory(absolute.parent, "stream artifact parent")
    try:
        absolute.lstat()
    except FileNotFoundError:
        absolute.mkdir(parents=False, exist_ok=False)
    return _canonical_artifact_directory(absolute, "stream artifact root")


def canonical_json_bytes(value: Any, *, trailing_lf: bool = True) -> bytes:
    """Return the single canonical JSON encoding used by all A4 hashes."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if trailing_lf else b"")


def _require_sha256(name: str, value: Any) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} is not a lowercase SHA256")
    return text


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def logical_row_count(
    *, num_layers: int, num_query_heads: int, answer_query_count: int, certified_parent_count: int
) -> int:
    """Return L*Hq*Q*P with no observation-derived logical-row ceiling."""

    factors = (
        _positive_integer("num_layers", num_layers),
        _positive_integer("num_query_heads", num_query_heads),
        _positive_integer("answer_query_count", answer_query_count),
        _positive_integer("certified_parent_count", certified_parent_count),
    )
    return math.prod(factors)


def encode_row_ordinal(
    *,
    layer: int,
    query_head: int,
    query_index: int,
    parent_index: int,
    num_layers: int,
    num_query_heads: int,
    answer_query_count: int,
    certified_parent_count: int,
) -> int:
    """Apply r=(((layer*Hq+head)*Q+query)*P+parent)."""

    total = logical_row_count(
        num_layers=num_layers,
        num_query_heads=num_query_heads,
        answer_query_count=answer_query_count,
        certified_parent_count=certified_parent_count,
    )
    coordinates = (
        ("layer", layer, num_layers),
        ("query_head", query_head, num_query_heads),
        ("query_index", query_index, answer_query_count),
        ("parent_index", parent_index, certified_parent_count),
    )
    for name, value, bound in coordinates:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < bound:
            raise ValueError(f"{name} is outside [0,{bound})")
    ordinal = (((layer * num_query_heads + query_head) * answer_query_count + query_index) * certified_parent_count + parent_index)
    if not 0 <= ordinal < total:  # defensive check against contract drift
        raise AssertionError("encoded ordinal is outside the logical universe")
    return ordinal


def decode_row_ordinal(
    row_ordinal: int,
    *,
    num_layers: int,
    num_query_heads: int,
    answer_query_count: int,
    certified_parent_count: int,
) -> tuple[int, int, int, int]:
    """Invert the A4 sample-local ordinal without constructing any rows."""

    total = logical_row_count(
        num_layers=num_layers,
        num_query_heads=num_query_heads,
        answer_query_count=answer_query_count,
        certified_parent_count=certified_parent_count,
    )
    if isinstance(row_ordinal, bool) or not isinstance(row_ordinal, int) or not 0 <= row_ordinal < total:
        raise ValueError(f"row_ordinal is outside [0,{total})")
    parent_index = row_ordinal % certified_parent_count
    value = row_ordinal // certified_parent_count
    query_index = value % answer_query_count
    value //= answer_query_count
    query_head = value % num_query_heads
    layer = value // num_query_heads
    if layer >= num_layers:
        raise AssertionError("decoded layer exceeds the frozen geometry")
    return layer, query_head, query_index, parent_index


def logical_row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    missing = [name for name in ROW_KEY_COLUMNS if name not in row]
    if missing:
        raise ValueError(f"logical row key is incomplete: {missing}")
    return tuple(row[name] for name in ROW_KEY_COLUMNS)


def logical_row_id(row: Mapping[str, Any]) -> str:
    """Hash domain NUL compact-canonical existing 15-field row-key array."""

    key_bytes = canonical_json_bytes(logical_row_key(row), trailing_lf=False)
    return hashlib.sha256(LOGICAL_ROW_ID_DOMAIN + b"\0" + key_bytes).hexdigest()


def canonical_endpoint_id(
    seed: Any,
    checkpoint_arm: Any,
    inference_operator: Any,
    cell: Any,
    task: Any,
    lambda_value: Any,
) -> str:
    """Encode the frozen six-field endpoint identity array, with no LF."""

    values = [seed, checkpoint_arm, inference_operator, cell, task, lambda_value]
    encoded = canonical_json_bytes(values, trailing_lf=False).decode("utf-8")
    return _validate_endpoint_id(encoded)


def _validate_endpoint_id(value: Any) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("endpoint_id must be a nonempty NUL-free UTF-8 string")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("endpoint_id is not canonical JSON") from error
    if not isinstance(decoded, list) or len(decoded) != len(ENDPOINT_ID_COLUMNS):
        raise ValueError("endpoint_id must be the frozen six-field JSON array")
    if isinstance(decoded[0], bool) or not isinstance(decoded[0], int) or decoded[0] < 0:
        raise ValueError("endpoint_id seed must be a nonnegative integer")
    if any(not isinstance(decoded[index], str) or not decoded[index] for index in range(1, 5)):
        raise ValueError("endpoint_id string fields must be nonempty")
    if isinstance(decoded[5], bool) or not isinstance(decoded[5], (int, float)) or not math.isfinite(decoded[5]):
        raise ValueError("endpoint_id lambda_value must be finite numeric")
    if canonical_json_bytes(decoded, trailing_lf=False).decode("utf-8") != value:
        raise ValueError("endpoint_id is not compact canonical JSON")
    return value


def endpoint_row_id(endpoint_id: str, logical_id: str) -> str:
    """Hash UTF8(endpoint_id) NUL ASCII(logical_row_id), exactly once."""

    endpoint_id = _validate_endpoint_id(endpoint_id)
    logical_id = _require_sha256("logical_row_id", logical_id)
    return hashlib.sha256(endpoint_id.encode("utf-8") + b"\0" + logical_id.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class ChunkRange:
    chunk_index: int
    first_row_ordinal: int
    end_row_ordinal_exclusive: int

    @property
    def row_count(self) -> int:
        return self.end_row_ordinal_exclusive - self.first_row_ordinal


def iter_chunk_ranges(expected_rows: int, chunk_rows: int = PHYSICAL_CHUNK_ROWS) -> Iterator[ChunkRange]:
    expected_rows = _positive_integer("expected_rows", expected_rows)
    chunk_rows = _positive_integer("chunk_rows", chunk_rows)
    for chunk_index, first in enumerate(range(0, expected_rows, chunk_rows)):
        yield ChunkRange(chunk_index, first, min(first + chunk_rows, expected_rows))


class SampleRowStream:
    """Random-access logical-row factory over compact frozen sample geometry."""

    def __init__(
        self,
        item: Mapping[str, Any],
        *,
        num_layers: int,
        num_query_heads: int,
        num_kv_heads: int,
        endpoint_id: str,
        schema_sha256: str,
        extra_row_factory: Callable[[int, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self.item = item
        self.num_layers = _positive_integer("num_layers", num_layers)
        self.num_query_heads = _positive_integer("num_query_heads", num_query_heads)
        self.num_kv_heads = _positive_integer("num_kv_heads", num_kv_heads)
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        self.endpoint_id = _validate_endpoint_id(endpoint_id)
        endpoint_parts = json.loads(self.endpoint_id)
        if "task" in item and item["task"] != endpoint_parts[4]:
            raise ValueError("endpoint_id task differs from the sample task")
        self.schema_sha256 = _require_sha256("schema_sha256", schema_sha256)
        self.extra_row_factory = extra_row_factory
        self.answer_queries = item.get("answer_queries")
        self.certified_parents = item.get("certified_parents")
        if not isinstance(self.answer_queries, Sequence) or not self.answer_queries:
            raise ValueError("answer_queries must be a nonempty frozen sequence")
        if not isinstance(self.certified_parents, Sequence) or not self.certified_parents:
            raise ValueError("certified_parents must be a nonempty frozen sequence")
        for name in (
            "sample_sha256",
            "content_group_sha256",
        ):
            _require_sha256(name, item.get(name))
        provenance = item.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("item provenance is missing")
        for name in ("input_sha256", "alignment_sha256", "labels_sha256", "gold_response_sha256"):
            _require_sha256(name, provenance.get(name))
        query_positions: set[int] = set()
        for query in self.answer_queries:
            if not isinstance(query, Mapping):
                raise ValueError("answer query is not a mapping")
            values = tuple(query.get(name) for name in ("query_position", "target_position", "target_token_id"))
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
                raise ValueError("answer query fields must be nonnegative integers")
            if values[0] in query_positions:
                raise ValueError("answer query positions are not unique")
            query_positions.add(values[0])
        parent_positions: set[int] = set()
        for parent in self.certified_parents:
            if not isinstance(parent, Mapping):
                raise ValueError("certified parent is not a mapping")
            position = parent.get("parent_position")
            count = parent.get("candidate_count")
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise ValueError("parent_position must be a nonnegative integer")
            if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 4:
                raise ValueError("certified candidate_count must be in [2,4]")
            if position in parent_positions:
                raise ValueError("certified parent positions are not unique")
            parent_positions.add(position)
            prior = parent.get("prior")
            if not isinstance(prior, Sequence) or len(prior) != count:
                raise ValueError("certified parent prior length differs from candidate_count")
            prior_values = tuple(float(value) for value in prior)
            if any(not math.isfinite(value) or value <= 0 for value in prior_values) or not math.isclose(sum(prior_values), 1.0, abs_tol=2e-5, rel_tol=0):
                raise ValueError("certified parent prior is invalid")
            if not isinstance(parent.get("topology"), str) or not parent["topology"]:
                raise ValueError("certified parent topology is missing")
            candidate_indices = parent.get("candidate_indices")
            candidate_valid = parent.get("candidate_valid_mask")
            slot_weights = parent.get("candidate_slot_weights")
            if not (
                isinstance(candidate_indices, Sequence)
                and isinstance(candidate_valid, Sequence)
                and isinstance(slot_weights, Sequence)
                and len(candidate_indices) == len(candidate_valid) == len(slot_weights) == 4
            ):
                raise ValueError("certified parent slot geometry must contain four frozen slots")
            normalized_indices = [int(value) for value in candidate_indices]
            normalized_valid = [bool(value) for value in candidate_valid]
            normalized_weights = [float(value) for value in slot_weights]
            expected_valid = [
                index >= 0 and math.isfinite(weight) and weight > 0
                for index, weight in zip(normalized_indices, normalized_weights)
            ]
            legal_indices = [
                index for index, valid in zip(normalized_indices, normalized_valid) if valid
            ]
            legal_weights = [
                weight for weight, valid in zip(normalized_weights, normalized_valid) if valid
            ]
            if (
                normalized_valid != expected_valid
                or len(legal_indices) != count
                or len(set(legal_indices)) != count
                or any(index < 0 for index in legal_indices)
                or any(
                    (not valid) and (index != -1 or weight != 0.0)
                    for index, weight, valid in zip(
                        normalized_indices, normalized_weights, normalized_valid
                    )
                )
                or any(
                    abs(left - right) > 2e-5
                    for left, right in zip(legal_weights, prior_values)
                )
            ):
                raise ValueError("certified parent candidate slot identity/prior changed")
            statistical_weight = float(parent.get("statistical_weight", 1.0))
            if not math.isfinite(statistical_weight) or statistical_weight <= 0:
                raise ValueError("certified parent statistical weight must be positive finite")

    @property
    def row_count(self) -> int:
        return logical_row_count(
            num_layers=self.num_layers,
            num_query_heads=self.num_query_heads,
            answer_query_count=len(self.answer_queries),
            certified_parent_count=len(self.certified_parents),
        )

    def row_at(self, row_ordinal: int) -> dict[str, Any]:
        layer, query_head, query_index, parent_index = decode_row_ordinal(
            row_ordinal,
            num_layers=self.num_layers,
            num_query_heads=self.num_query_heads,
            answer_query_count=len(self.answer_queries),
            certified_parent_count=len(self.certified_parents),
        )
        query = self.answer_queries[query_index]
        parent = self.certified_parents[parent_index]
        provenance = self.item["provenance"]
        row: dict[str, Any] = {
            "sample_sha256": self.item["sample_sha256"],
            "content_group_sha256": self.item["content_group_sha256"],
            "input_sha256": provenance["input_sha256"],
            "alignment_sha256": provenance["alignment_sha256"],
            "labels_sha256": provenance["labels_sha256"],
            "gold_response_sha256": provenance["gold_response_sha256"],
            "layer": layer,
            "query_head": query_head,
            "kv_head": query_head // (self.num_query_heads // self.num_kv_heads),
            "query_position": query["query_position"],
            "target_position": query["target_position"],
            "target_token_id": query["target_token_id"],
            "parent_position": parent["parent_position"],
            "candidate_count": parent["candidate_count"],
            "topology": parent["topology"],
            "prior": [float(value) for value in parent["prior"]],
            "candidate_indices": [int(value) for value in parent["candidate_indices"]],
            "candidate_valid_mask": [bool(value) for value in parent["candidate_valid_mask"]],
            "candidate_slot_weights": [float(value) for value in parent["candidate_slot_weights"]],
            "statistical_weight": float(parent.get("statistical_weight", 1.0)),
            "row_ordinal": row_ordinal,
            "endpoint_id": self.endpoint_id,
            "schema_sha256": self.schema_sha256,
        }
        row["logical_row_id"] = logical_row_id(row)
        row["endpoint_row_id"] = endpoint_row_id(self.endpoint_id, row["logical_row_id"])
        if self.extra_row_factory is not None:
            extra = self.extra_row_factory(row_ordinal, row)
            if not isinstance(extra, Mapping):
                raise ValueError("extra_row_factory must return a mapping")
            overlap = PROTECTED_STREAM_COLUMNS.intersection(extra)
            if overlap:
                raise ValueError(f"extra row values overwrite protected fields: {sorted(overlap)}")
            row.update(extra)
        canonical_json_bytes(row)  # fail immediately on NaN/Inf or unserializable values
        return row

    def iter_rows(self, first: int = 0, end: int | None = None) -> Iterator[dict[str, Any]]:
        if end is None:
            end = self.row_count
        if isinstance(first, bool) or isinstance(end, bool) or not isinstance(first, int) or not isinstance(end, int):
            raise ValueError("stream bounds must be integers")
        if not 0 <= first <= end <= self.row_count:
            raise ValueError("stream bounds are outside the logical universe")
        for ordinal in range(first, end):
            yield self.row_at(ordinal)


def semantic_row_bytes(row: Mapping[str, Any]) -> bytes:
    """Canonical semantic bytes; the LF is part of the stream framing."""

    return canonical_json_bytes(row, trailing_lf=True)


def attest_ordinal_payload_stream(
    records: Iterable[tuple[int, bytes]],
    *,
    expected_rows: int,
    chunk_rows: int = PHYSICAL_CHUNK_ROWS,
) -> dict[str, Any]:
    """Attest canonical LF-framed row bytes with O(1) live records.

    This deliberately uses the exact semantic-stream hash algorithm: canonical
    semantic row bytes (already LF framed) are concatenated directly in ordinal
    order.  There is no stress-only length prefix or alternate hash domain.
    """

    expected_rows = _positive_integer("expected_rows", expected_rows)
    chunk_rows = _positive_integer("chunk_rows", chunk_rows)
    digest = hashlib.sha256()
    count = 0
    chunk_count = 0
    rows_in_chunk = 0
    max_rows_in_chunk = 0
    for ordinal, payload in records:
        if ordinal != count:
            raise ValueError("ordinal payload stream has a gap, duplicate, or reorder")
        if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
            raise ValueError("semantic payload must be LF-framed bytes")
        digest.update(payload)
        count += 1
        rows_in_chunk += 1
        if rows_in_chunk == chunk_rows or count == expected_rows:
            chunk_count += 1
            max_rows_in_chunk = max(max_rows_in_chunk, rows_in_chunk)
            rows_in_chunk = 0
    if count != expected_rows:
        raise ValueError(f"semantic payload count differs: {count} != {expected_rows}")
    return {
        "count": count,
        "first_row_ordinal": 0,
        "last_row_ordinal": count - 1,
        "semantic_stream_sha256": digest.hexdigest(),
        "physical_chunk_rows": chunk_rows,
        "chunk_count": chunk_count,
        "max_rows_in_chunk": max_rows_in_chunk,
    }


def attest_ordered_sample_streams(streams: Iterable[SampleRowStream]) -> dict[str, Any]:
    """Hash sample streams in the frozen (sample_sha256,row_ordinal) order."""

    digest = hashlib.sha256()
    previous_sample: str | None = None
    sample_count = 0
    row_count = 0
    first_sample: str | None = None
    endpoint: str | None = None
    schema: str | None = None
    for stream in streams:
        sample = _require_sha256("sample_sha256", stream.item["sample_sha256"])
        if previous_sample is not None and sample <= previous_sample:
            raise ValueError("sample streams are duplicate or not in SHA order")
        if endpoint is None:
            endpoint = stream.endpoint_id
            schema = stream.schema_sha256
            first_sample = sample
        elif stream.endpoint_id != endpoint or stream.schema_sha256 != schema:
            raise ValueError("ordered sample streams mix endpoint or schema identities")
        for row in stream.iter_rows():
            digest.update(semantic_row_bytes(row))
            row_count += 1
        previous_sample = sample
        sample_count += 1
    if sample_count == 0:
        raise ValueError("ordered sample stream universe is empty")
    return {
        "sample_count": sample_count,
        "logical_row_count": row_count,
        "first_sample_sha256": first_sample,
        "last_sample_sha256": previous_sample,
        "endpoint_id": endpoint,
        "schema_sha256": schema,
        "semantic_stream_sha256": digest.hexdigest(),
    }


def _invoke_failure(
    failure_injector: Callable[[str, int | None], None] | None,
    stage: str,
    chunk_index: int | None,
) -> None:
    if stage not in CRASH_STAGES:
        raise AssertionError(f"unknown crash stage: {stage}")
    if failure_injector is not None:
        failure_injector(stage, chunk_index)


def _publish_no_overwrite(temporary: Path, final: Path) -> None:
    """Atomically publish a same-filesystem file without replacing a winner."""

    try:
        os.link(temporary, final)
    except FileExistsError:
        raise FileExistsError(f"immutable stream artifact already exists: {final}")
    else:
        temporary.unlink()
        descriptor = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _chunk_filename(chunk_index: int) -> str:
    return f"chunk_{chunk_index:08d}.jsonl"


def _write_one_chunk(
    root: Path,
    stream: SampleRowStream,
    interval: ChunkRange,
    failure_injector: Callable[[str, int | None], None] | None,
) -> None:
    final = root / _chunk_filename(interval.chunk_index)
    if final.exists():
        return
    temporary = root / f".{final.name}.tmp"
    if temporary.exists():
        temporary.unlink()  # stale crash output is never a valid immutable chunk
    halfway = interval.first_row_ordinal + max(1, (interval.row_count + 1) // 2)
    halfway_called = False
    with temporary.open("xb") as handle:
        for ordinal in range(interval.first_row_ordinal, interval.end_row_ordinal_exclusive):
            handle.write(semantic_row_bytes(stream.row_at(ordinal)))
            if not halfway_called and ordinal + 1 >= halfway:
                halfway_called = True
                handle.flush()
                _invoke_failure(failure_injector, "chunk_temporary_half_written", interval.chunk_index)
        _invoke_failure(failure_injector, "before_chunk_close", interval.chunk_index)
        handle.flush()
        os.fsync(handle.fileno())
    _invoke_failure(failure_injector, "before_artifact_rename", interval.chunk_index)
    _publish_no_overwrite(temporary, final)


def _scan_chunk(
    path: Path,
    stream: SampleRowStream,
    interval: ChunkRange,
    *,
    full_semantic_digest: Any | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing immutable stream chunk: {path.name}")
    physical_digest = hashlib.sha256()
    semantic_digest = hashlib.sha256()
    physical_bytes = 0
    count = 0
    first_id: str | None = None
    last_id: str | None = None
    with path.open("rb") as handle:
        for ordinal in range(interval.first_row_ordinal, interval.end_row_ordinal_exclusive):
            payload = handle.readline()
            if not payload:
                raise ValueError("stream chunk ended before its ordinal range")
            physical_digest.update(payload)
            physical_bytes += len(payload)
            try:
                row = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("stream chunk contains invalid canonical JSON") from error
            if not isinstance(row, dict):
                raise ValueError("stream chunk row is not an object")
            expected = stream.row_at(ordinal)
            if row != expected:
                raise ValueError("stream row mutation, reorder, identity, endpoint, or schema mismatch")
            canonical = semantic_row_bytes(row)
            if payload != canonical:
                raise ValueError("stream row physical encoding is not canonical")
            semantic_digest.update(canonical)
            if full_semantic_digest is not None:
                full_semantic_digest.update(canonical)
            logical_id = _require_sha256("logical_row_id", row.get("logical_row_id"))
            _require_sha256("endpoint_row_id", row.get("endpoint_row_id"))
            if first_id is None:
                first_id = logical_id
            last_id = logical_id
            count += 1
        if handle.read(1):
            raise ValueError("stream chunk contains rows beyond its ordinal range")
    if count != interval.row_count or first_id is None or last_id is None:
        raise ValueError("stream chunk row count is inconsistent")
    return {
        "sample_sha256": stream.item["sample_sha256"],
        "endpoint_id": stream.endpoint_id,
        "chunk_index": interval.chunk_index,
        "first_row_ordinal": interval.first_row_ordinal,
        "end_row_ordinal_exclusive": interval.end_row_ordinal_exclusive,
        "row_count": count,
        "first_logical_row_id": first_id,
        "last_logical_row_id": last_id,
        "canonical_row_stream_sha256": semantic_digest.hexdigest(),
        "physical_file": path.name,
        "physical_file_sha256": physical_digest.hexdigest(),
        "physical_file_bytes": physical_bytes,
        "schema_sha256": stream.schema_sha256,
    }


def _validate_manifest_intervals(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    expected_rows = _positive_integer("expected_logical_rows", manifest.get("expected_logical_rows"))
    chunk_rows = _positive_integer("physical_chunk_rows", manifest.get("physical_chunk_rows"))
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("stream manifest has no chunk records")
    expected_chunk_count = math.ceil(expected_rows / chunk_rows)
    if len(chunks) != expected_chunk_count:
        raise ValueError("stream manifest has a missing or duplicate chunk")
    filenames: set[str] = set()
    previous_end = 0
    total = 0
    for expected_index, record in enumerate(chunks):
        if not isinstance(record, Mapping):
            raise ValueError("stream chunk record is not an object")
        if record.get("chunk_index") != expected_index:
            raise ValueError("stream chunk indices are duplicate, missing, or reordered")
        first = record.get("first_row_ordinal")
        end = record.get("end_row_ordinal_exclusive")
        count = record.get("row_count")
        expected_first = expected_index * chunk_rows
        expected_end = min(expected_first + chunk_rows, expected_rows)
        if first != expected_first or end != expected_end:
            if first != previous_end:
                raise ValueError("stream chunk ranges overlap or contain an ordinal gap")
            raise ValueError("stream chunk range differs from deterministic partition")
        if count != end - first or count <= 0:
            raise ValueError("stream chunk row count differs from its range")
        filename = record.get("physical_file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("stream chunk physical path is not a safe basename")
        if filename in filenames:
            raise ValueError("stream manifest references a duplicate physical chunk")
        filenames.add(filename)
        previous_end = end
        total += count
    if previous_end != expected_rows or total != expected_rows:
        raise ValueError("stream chunk ranges do not exactly cover [0,N)")
    return chunks


def _load_manifest(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("stream manifest is not an object")
    return value


def validate_streaming_schema_artifact(
    value: Mapping[str, Any], schema_path: Path = DEFAULT_STREAMING_SCHEMA_PATH
) -> None:
    """Validate one complete A4 artifact against the tracked v6 meta-schema."""
    schema = _load_manifest(schema_path)
    _validate_json_schema(dict(value), schema, schema, path="$")


def _validate_json_schema(
    value: Any,
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    *,
    path: str,
) -> None:
    """Validate the strict JSON-Schema subset used by the tracked A4 schema.

    The project environment intentionally has no optional ``jsonschema``
    dependency.  This fail-closed validator implements every keyword present in
    ``e1_streaming_schema.json`` and rejects unsupported local references.
    """

    if "$ref" in schema:
        reference = schema["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
            raise ValueError(f"{path}: unsupported JSON schema reference")
        name = reference.removeprefix("#/$defs/")
        target = root.get("$defs", {}).get(name)
        if not isinstance(target, Mapping):
            raise ValueError(f"{path}: unresolved JSON schema reference {reference}")
        _validate_json_schema(value, target, root, path=path)
        return
    if "oneOf" in schema:
        matches = 0
        errors = []
        for branch in schema["oneOf"]:
            try:
                _validate_json_schema(value, branch, root, path=path)
            except ValueError as error:
                errors.append(str(error))
            else:
                matches += 1
        if matches != 1:
            raise ValueError(f"{path}: JSON schema oneOf matched {matches} branches")
        return
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: value differs from JSON schema const")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value is outside JSON schema enum")
    expected_type = schema.get("type")
    type_ok = {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
    }.get(expected_type, True)
    if expected_type is not None and not type_ok:
        raise ValueError(f"{path}: value differs from JSON schema type {expected_type}")
    if isinstance(value, Mapping):
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(f"{path}: JSON schema required properties missing: {missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value).difference(properties))
            if extra:
                raise ValueError(f"{path}: JSON schema additional properties: {extra}")
        for name, child in value.items():
            child_schema = properties.get(name)
            if isinstance(child_schema, Mapping):
                _validate_json_schema(child, child_schema, root, path=f"{path}.{name}")
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            raise ValueError(f"{path}: JSON schema minItems failed")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ValueError(f"{path}: JSON schema maxItems failed")
        if schema.get("uniqueItems") and len(
            {canonical_json_bytes(item, trailing_lf=False) for item in value}
        ) != len(value):
            raise ValueError(f"{path}: JSON schema uniqueItems failed")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, child in enumerate(value):
                _validate_json_schema(child, item_schema, root, path=f"{path}[{index}]")
        contains = schema.get("contains")
        if isinstance(contains, Mapping):
            matched = False
            for index, child in enumerate(value):
                try:
                    _validate_json_schema(child, contains, root, path=f"{path}[{index}]")
                except ValueError:
                    continue
                matched = True
                break
            if not matched:
                raise ValueError(f"{path}: JSON schema contains failed")
    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            raise ValueError(f"{path}: JSON schema minLength failed")
        if "pattern" in schema and re.search(str(schema["pattern"]), value) is None:
            raise ValueError(f"{path}: JSON schema pattern failed")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path}: JSON schema numeric value is nonfinite")
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path}: JSON schema minimum failed")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path}: JSON schema maximum failed")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ValueError(f"{path}: JSON schema exclusiveMinimum failed")


def verify_stream_artifact(manifest_path: Path, stream: SampleRowStream) -> dict[str, Any]:
    """Verify ranges, physical bytes, identities, and the full semantic stream."""

    manifest = _load_manifest(manifest_path)
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("stream manifest protocol/schema version mismatch")
    if manifest.get("status") != "COMPLETE_IMMUTABLE":
        raise ValueError("stream manifest is not complete")
    if manifest.get("sample_sha256") != stream.item["sample_sha256"]:
        raise ValueError("stream manifest sample identity mismatch")
    if manifest.get("endpoint_id") != stream.endpoint_id:
        raise ValueError("stream manifest endpoint identity mismatch")
    if manifest.get("schema_sha256") != stream.schema_sha256:
        raise ValueError("stream manifest physical schema mismatch")
    if manifest.get("expected_logical_rows") != stream.row_count:
        raise ValueError("stream manifest expected N_s mismatch")
    chunks = _validate_manifest_intervals(manifest)
    full_digest = hashlib.sha256()
    observed_count = 0
    for record in chunks:
        interval = ChunkRange(
            int(record["chunk_index"]),
            int(record["first_row_ordinal"]),
            int(record["end_row_ordinal_exclusive"]),
        )
        actual = _scan_chunk(
            manifest_path.parent / str(record["physical_file"]),
            stream,
            interval,
            full_semantic_digest=full_digest,
        )
        if actual != dict(record):
            raise ValueError("stream chunk manifest SHA/size/range/identity mismatch")
        observed_count += actual["row_count"]
    semantic_sha = full_digest.hexdigest()
    if semantic_sha != manifest.get("semantic_stream_sha256"):
        raise ValueError("stream manifest semantic stream SHA mismatch")
    if observed_count != stream.row_count:
        raise ValueError("stream emitted row count differs from expected N_s")
    return {
        "status": "GO",
        "expected_logical_rows": stream.row_count,
        "emitted_logical_rows": observed_count,
        "ordinal_first": 0,
        "ordinal_last": observed_count - 1,
        "semantic_stream_sha256": semantic_sha,
        "chunk_count": len(chunks),
    }


def write_stream_artifact(
    output_root: Path,
    stream: SampleRowStream,
    *,
    chunk_rows: int = PHYSICAL_CHUNK_ROWS,
    failure_injector: Callable[[str, int | None], None] | None = None,
) -> dict[str, Any]:
    """Write or resume immutable chunks, then atomically publish one manifest."""

    chunk_rows = _positive_integer("chunk_rows", chunk_rows)
    output_root = _prepare_artifact_root(output_root)
    manifest_path = output_root / MANIFEST_NAME
    intervals = list(iter_chunk_ranges(stream.row_count, chunk_rows))
    preflight_paths = [
        manifest_path,
        output_root / f".{MANIFEST_NAME}.tmp",
        *(
            path
            for interval in intervals
            for path in (
                output_root / _chunk_filename(interval.chunk_index),
                output_root / f".{_chunk_filename(interval.chunk_index)}.tmp",
            )
        ),
    ]
    for path in preflight_paths:
        _preflight_artifact_file(path, "stream producer final/temporary")
    if manifest_path.exists():
        return verify_stream_artifact(manifest_path, stream)
    chunk_records: list[dict[str, Any]] = []
    full_digest = hashlib.sha256()
    for interval in intervals:
        _write_one_chunk(output_root, stream, interval, failure_injector)
        record = _scan_chunk(
            output_root / _chunk_filename(interval.chunk_index),
            stream,
            interval,
            full_semantic_digest=full_digest,
        )
        chunk_records.append(record)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_IMMUTABLE",
        "sample_sha256": stream.item["sample_sha256"],
        "endpoint_id": stream.endpoint_id,
        "schema_sha256": stream.schema_sha256,
        "physical_format": "canonical_jsonl_reference_only",
        "physical_chunk_rows": chunk_rows,
        "expected_logical_rows": stream.row_count,
        "semantic_stream_sha256": full_digest.hexdigest(),
        "chunks": chunk_records,
    }
    _invoke_failure(failure_injector, "before_manifest_write", None)
    temporary = output_root / f".{MANIFEST_NAME}.tmp"
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(manifest))
        handle.flush()
        os.fsync(handle.fileno())
    _invoke_failure(failure_injector, "before_manifest_rename", None)
    _publish_no_overwrite(temporary, manifest_path)
    return verify_stream_artifact(manifest_path, stream)


def _parquet_chunk_filename(chunk_index: int) -> str:
    return f"chunk_{chunk_index:08d}.parquet"


def _bounded_rows(stream: SampleRowStream, interval: ChunkRange) -> list[dict[str, Any]]:
    """Materialize exactly one bounded physical chunk, never a sample table."""

    if interval.row_count > PHYSICAL_CHUNK_ROWS:
        raise ValueError("a parquet writer buffer exceeds production_chunk_rows=4096")
    return [
        stream.row_at(ordinal)
        for ordinal in range(interval.first_row_ordinal, interval.end_row_ordinal_exclusive)
    ]


def _physical_file_attestation(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while True:
            payload = handle.read(1024 * 1024)
            if not payload:
                break
            digest.update(payload)
            byte_count += len(payload)
    if byte_count <= 0:
        raise ValueError("physical parquet chunk is empty")
    return byte_count, digest.hexdigest()


def _iter_parquet_records(path: Path, *, batch_rows: int = PHYSICAL_CHUNK_ROWS) -> Iterator[dict[str, Any]]:
    """Decode bounded record batches without read_table or batch.to_pylist."""

    import pyarrow.parquet as parquet

    parquet_file = parquet.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_rows):
        names = batch.schema.names
        for row_index in range(batch.num_rows):
            yield {
                name: batch.column(column_index)[row_index].as_py()
                for column_index, name in enumerate(names)
            }


def _parquet_schema(path: Path) -> Any:
    import pyarrow.parquet as parquet

    return parquet.ParquetFile(path).schema_arrow


def _arrow_type_signature(data_type: Any) -> Any:
    """Compare physical Arrow types while ignoring Parquet list-child aliases."""

    import pyarrow as arrow

    if arrow.types.is_list(data_type):
        return ("list", _arrow_type_signature(data_type.value_type))
    if arrow.types.is_large_list(data_type):
        return ("large_list", _arrow_type_signature(data_type.value_type))
    if arrow.types.is_fixed_size_list(data_type):
        return (
            "fixed_size_list",
            data_type.list_size,
            _arrow_type_signature(data_type.value_type),
        )
    if arrow.types.is_struct(data_type):
        return (
            "struct",
            tuple(
                (field.name, field.nullable, _arrow_type_signature(field.type))
                for field in data_type
            ),
        )
    return str(data_type)


def _arrow_schema_signature(schema: Any) -> tuple[Any, ...]:
    return tuple(
        (field.name, field.nullable, _arrow_type_signature(field.type))
        for field in schema
    )


def _write_one_parquet_chunk(
    root: Path,
    stream: SampleRowStream,
    interval: ChunkRange,
    arrow_schema: Any,
    failure_injector: Callable[[str, int | None], None] | None,
) -> None:
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    final = root / _parquet_chunk_filename(interval.chunk_index)
    if final.exists():
        return
    temporary = root / f".{final.name}.tmp"
    if temporary.exists():
        temporary.unlink()
    rows = _bounded_rows(stream, interval)
    table = arrow.Table.from_pylist(rows, schema=arrow_schema)
    writer = parquet.ParquetWriter(temporary, arrow_schema)
    try:
        if failure_injector is None:
            writer.write_table(table, row_group_size=PHYSICAL_CHUNK_ROWS)
        else:
            halfway = max(1, (table.num_rows + 1) // 2)
            writer.write_table(table.slice(0, halfway), row_group_size=PHYSICAL_CHUNK_ROWS)
            _invoke_failure(
                failure_injector,
                "chunk_temporary_half_written",
                interval.chunk_index,
            )
            if halfway < table.num_rows:
                writer.write_table(
                    table.slice(halfway), row_group_size=PHYSICAL_CHUNK_ROWS
                )
        _invoke_failure(failure_injector, "before_chunk_close", interval.chunk_index)
        writer.close()
    except BaseException:
        # A real crash might leave an unclosed footer.  Closing here only makes
        # the synthetic temporary easier for the filesystem to release; .tmp
        # is never accepted as a completed artifact in either case.
        try:
            writer.close()
        except BaseException:
            pass
        raise
    _invoke_failure(failure_injector, "before_artifact_rename", interval.chunk_index)
    _publish_no_overwrite(temporary, final)


def _scan_parquet_chunk(
    path: Path,
    stream: SampleRowStream,
    interval: ChunkRange,
    arrow_schema: Any,
    *,
    full_semantic_digest: Any | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing immutable parquet chunk: {path.name}")
    if _arrow_schema_signature(_parquet_schema(path)) != _arrow_schema_signature(
        arrow_schema
    ):
        raise ValueError("parquet chunk physical schema mismatch")
    semantic_digest = hashlib.sha256()
    count = 0
    first_id: str | None = None
    last_id: str | None = None
    for row in _iter_parquet_records(path):
        ordinal = interval.first_row_ordinal + count
        if ordinal >= interval.end_row_ordinal_exclusive:
            raise ValueError("parquet chunk contains rows beyond its ordinal range")
        expected = stream.row_at(ordinal)
        if row != expected:
            raise ValueError(
                "parquet row mutation, reorder, identity, endpoint, or schema mismatch"
            )
        canonical = semantic_row_bytes(row)
        semantic_digest.update(canonical)
        if full_semantic_digest is not None:
            full_semantic_digest.update(canonical)
        logical_id = _require_sha256("logical_row_id", row.get("logical_row_id"))
        _require_sha256("endpoint_row_id", row.get("endpoint_row_id"))
        if first_id is None:
            first_id = logical_id
        last_id = logical_id
        count += 1
    if count != interval.row_count or first_id is None or last_id is None:
        raise ValueError("parquet chunk row count differs from its ordinal range")
    physical_bytes, physical_sha = _physical_file_attestation(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "sample_sha256": stream.item["sample_sha256"],
        "chunk_index": interval.chunk_index,
        "first_row_ordinal": interval.first_row_ordinal,
        "end_row_ordinal_exclusive": interval.end_row_ordinal_exclusive,
        "row_count": count,
        "first_logical_row_id": first_id,
        "last_logical_row_id": last_id,
        "canonical_row_stream_sha256": semantic_digest.hexdigest(),
        "relative_path": path.name,
        "physical_file_sha256": physical_sha,
        "physical_file_bytes": physical_bytes,
        "schema_sha256": stream.schema_sha256,
    }


def _validate_parquet_manifest_intervals(
    manifest: Mapping[str, Any], stream: SampleRowStream
) -> list[Mapping[str, Any]]:
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("parquet manifest protocol/schema version mismatch")
    if manifest.get("artifact_type") != "chunk_manifest":
        raise ValueError("parquet manifest artifact type mismatch")
    if manifest.get("sample_sha256") != stream.item["sample_sha256"]:
        raise ValueError("parquet manifest sample identity mismatch")
    if manifest.get("expected_logical_rows") != stream.row_count:
        raise ValueError("parquet manifest expected N_s mismatch")
    expected_rows = _positive_integer(
        "expected_logical_rows", manifest.get("expected_logical_rows")
    )
    chunk_rows = _positive_integer(
        "physical_chunk_rows", manifest.get("physical_chunk_rows")
    )
    for name, expected in (
        ("missing_rows", 0),
        ("duplicate_rows", 0),
        ("overlapping_chunks", 0),
        ("ordinal_ranges_contiguous", True),
    ):
        if manifest.get(name) != expected:
            raise ValueError(f"parquet manifest integrity flag differs: {name}")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != math.ceil(expected_rows / chunk_rows):
        raise ValueError("parquet manifest has a missing or duplicate chunk")
    previous_end = 0
    total = 0
    relative_paths: set[str] = set()
    for expected_index, record in enumerate(chunks):
        if not isinstance(record, Mapping):
            raise ValueError("parquet chunk record is not an object")
        if record.get("schema_version") != SCHEMA_VERSION or record.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("parquet chunk record protocol/schema mismatch")
        if record.get("sample_sha256") != stream.item["sample_sha256"]:
            raise ValueError("parquet chunk record sample mismatch")
        if record.get("schema_sha256") != stream.schema_sha256:
            raise ValueError("parquet chunk record schema SHA mismatch")
        if record.get("chunk_index") != expected_index:
            raise ValueError("parquet chunk indices are duplicate, missing, or reordered")
        first = record.get("first_row_ordinal")
        end = record.get("end_row_ordinal_exclusive")
        count = record.get("row_count")
        expected_first = expected_index * chunk_rows
        expected_end = min(expected_first + chunk_rows, expected_rows)
        if first != expected_first or end != expected_end:
            if first != previous_end:
                raise ValueError("parquet ordinal ranges overlap or contain a gap")
            raise ValueError("parquet range differs from deterministic partition")
        if count != end - first or count <= 0 or count > PHYSICAL_CHUNK_ROWS:
            raise ValueError("parquet chunk row count differs from its bounded range")
        relative_path = record.get("relative_path")
        if not isinstance(relative_path, str):
            raise ValueError("parquet chunk relative path is missing")
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
            raise ValueError("parquet chunk relative path escapes its artifact root")
        if relative_path in relative_paths:
            raise ValueError("parquet manifest references a duplicate physical file")
        relative_paths.add(relative_path)
        previous_end = end
        total += count
    if previous_end != expected_rows or total != expected_rows:
        raise ValueError("parquet chunk ranges do not exactly cover [0,N)")
    return chunks


def verify_parquet_stream_artifact(
    manifest_path: Path, stream: SampleRowStream
) -> dict[str, Any]:
    """Replay a physical parquet stream without whole-table materialization."""

    manifest = _load_manifest(manifest_path)
    chunks = _validate_parquet_manifest_intervals(manifest, stream)
    referenced_chunks = {str(record["relative_path"]) for record in chunks}
    observed_chunks = {path.name for path in manifest_path.parent.glob("chunk_*.parquet")}
    temporary_files = {
        path.name for path in manifest_path.parent.iterdir()
        if path.is_file() and (path.name.startswith(".") or path.suffix == ".tmp")
    }
    if observed_chunks != referenced_chunks or temporary_files:
        raise ValueError("parquet artifact contains orphan, duplicate, or temporary chunks")
    first_path = manifest_path.parent / str(chunks[0]["relative_path"])
    arrow_schema = _parquet_schema(first_path)
    full_digest = hashlib.sha256()
    observed_count = 0
    for record in chunks:
        interval = ChunkRange(
            int(record["chunk_index"]),
            int(record["first_row_ordinal"]),
            int(record["end_row_ordinal_exclusive"]),
        )
        actual = _scan_parquet_chunk(
            manifest_path.parent / str(record["relative_path"]),
            stream,
            interval,
            arrow_schema,
            full_semantic_digest=full_digest,
        )
        if actual != dict(record):
            raise ValueError("parquet chunk manifest SHA/size/range/identity mismatch")
        observed_count += actual["row_count"]
    semantic_sha = full_digest.hexdigest()
    if semantic_sha != manifest.get("semantic_stream_sha256"):
        raise ValueError("parquet manifest semantic stream SHA mismatch")
    if observed_count != stream.row_count:
        raise ValueError("parquet emitted row count differs from expected N_s")
    return {
        "status": "GO",
        "expected_logical_rows": stream.row_count,
        "emitted_logical_rows": observed_count,
        "ordinal_first": 0,
        "ordinal_last": observed_count - 1,
        "semantic_stream_sha256": semantic_sha,
        "chunk_count": len(chunks),
    }


def write_parquet_stream_artifact(
    output_root: Path,
    stream: SampleRowStream,
    *,
    chunk_rows: int = PHYSICAL_CHUNK_ROWS,
    synthetic_partition_test: bool = False,
    failure_injector: Callable[[str, int | None], None] | None = None,
) -> dict[str, Any]:
    """Write immutable bounded parquet chunks and their v6 manifest.

    Production callers cannot select a non-4096 partition.  Smaller (or larger)
    boundaries are exposed only for preregistered synthetic equivalence tests.
    Even synthetic buffers remain capped at 4096 physical rows.
    """

    logical_partition_rows = _positive_integer("chunk_rows", chunk_rows)
    if logical_partition_rows != PHYSICAL_CHUNK_ROWS and not synthetic_partition_test:
        raise ValueError("production parquet physical_chunk_rows must equal 4096")
    physical_chunk_rows = min(logical_partition_rows, PHYSICAL_CHUNK_ROWS)
    output_root = _prepare_artifact_root(output_root)
    manifest_path = output_root / PARQUET_MANIFEST_NAME
    interval_list = list(iter_chunk_ranges(stream.row_count, physical_chunk_rows))
    preflight_paths = [
        manifest_path,
        output_root / f".{PARQUET_MANIFEST_NAME}.tmp",
        *(
            path
            for interval in interval_list
            for path in (
                output_root / _parquet_chunk_filename(interval.chunk_index),
                output_root
                / f".{_parquet_chunk_filename(interval.chunk_index)}.tmp",
            )
        ),
    ]
    for path in preflight_paths:
        _preflight_artifact_file(path, "parquet producer final/temporary")
    if manifest_path.exists():
        return verify_parquet_stream_artifact(manifest_path, stream)

    import pyarrow as arrow

    intervals = iter(interval_list)
    first_interval = next(intervals)
    first_path = output_root / _parquet_chunk_filename(first_interval.chunk_index)
    if first_path.exists():
        arrow_schema = _parquet_schema(first_path)
    else:
        arrow_schema = arrow.Table.from_pylist(
            _bounded_rows(stream, first_interval)
        ).schema
    chunk_records: list[dict[str, Any]] = []
    full_digest = hashlib.sha256()
    for interval in chain((first_interval,), intervals):
        _write_one_parquet_chunk(
            output_root, stream, interval, arrow_schema, failure_injector
        )
        record = _scan_parquet_chunk(
            output_root / _parquet_chunk_filename(interval.chunk_index),
            stream,
            interval,
            arrow_schema,
            full_semantic_digest=full_digest,
        )
        chunk_records.append(record)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "chunk_manifest",
        "sample_sha256": stream.item["sample_sha256"],
        "expected_logical_rows": stream.row_count,
        "physical_chunk_rows": physical_chunk_rows,
        "chunks": chunk_records,
        "semantic_stream_sha256": full_digest.hexdigest(),
        "missing_rows": 0,
        "duplicate_rows": 0,
        "overlapping_chunks": 0,
        "ordinal_ranges_contiguous": True,
    }
    if synthetic_partition_test and logical_partition_rows != physical_chunk_rows:
        manifest["synthetic_logical_partition_rows"] = logical_partition_rows
    _invoke_failure(failure_injector, "before_manifest_write", None)
    temporary = output_root / f".{PARQUET_MANIFEST_NAME}.tmp"
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(manifest))
        handle.flush()
        os.fsync(handle.fileno())
    _invoke_failure(failure_injector, "before_manifest_rename", None)
    _publish_no_overwrite(temporary, manifest_path)
    return verify_parquet_stream_artifact(manifest_path, stream)


def iter_verified_parquet_rows(
    manifest_path: Path, stream: SampleRowStream
) -> Iterator[dict[str, Any]]:
    """Yield verified parquet rows in frozen ordinal order by record batch."""

    verify_parquet_stream_artifact(manifest_path, stream)
    manifest = _load_manifest(manifest_path)
    for record in _validate_parquet_manifest_intervals(manifest, stream):
        path = manifest_path.parent / str(record["relative_path"])
        first = int(record["first_row_ordinal"])
        for offset, row in enumerate(_iter_parquet_records(path)):
            if row != stream.row_at(first + offset):
                raise ValueError("verified parquet stream changed during reread")
            yield row


def iter_verified_rows(manifest_path: Path, stream: SampleRowStream) -> Iterator[dict[str, Any]]:
    """Yield verified rows in sample-local ordinal order without a table read."""

    verify_stream_artifact(manifest_path, stream)
    manifest = _load_manifest(manifest_path)
    for record in _validate_manifest_intervals(manifest):
        path = manifest_path.parent / str(record["physical_file"])
        with path.open("rb") as handle:
            for ordinal in range(int(record["first_row_ordinal"]), int(record["end_row_ordinal_exclusive"])):
                payload = handle.readline()
                if not payload:
                    raise ValueError("verified stream changed during ordered reread")
                row = json.loads(payload)
                if row != stream.row_at(ordinal):
                    raise ValueError("verified stream changed during ordered reread")
                yield row
            if handle.read(1):
                raise ValueError("verified stream grew during ordered reread")


def ordered_numeric_aggregate(
    manifest_path: Path, stream: SampleRowStream, fields: Sequence[str]
) -> dict[str, Any]:
    """Reduce only in frozen ordinal order, independent of chunk boundaries."""

    if not fields or len(set(fields)) != len(fields):
        raise ValueError("aggregate fields must be a nonempty unique sequence")
    totals = {name: 0.0 for name in fields}
    count = 0
    for row in iter_verified_rows(manifest_path, stream):
        for name in fields:
            value = float(row[name])
            if not math.isfinite(value):
                raise ValueError(f"aggregate field {name} is nonfinite")
            totals[name] += value
        count += 1
    return {
        "row_count": count,
        "fields": {
            name: {"sum": totals[name], "mean": totals[name] / count}
            for name in fields
        },
    }


__all__ = [
    "CRASH_STAGES",
    "ENDPOINT_ID_COLUMNS",
    "LOGICAL_ROW_ID_DOMAIN",
    "MANIFEST_NAME",
    "PARQUET_MANIFEST_NAME",
    "PHYSICAL_CHUNK_ROWS",
    "PROTOCOL_ID",
    "ROW_KEY_COLUMNS",
    "SCHEMA_VERSION",
    "ChunkRange",
    "SampleRowStream",
    "attest_ordered_sample_streams",
    "attest_ordinal_payload_stream",
    "canonical_json_bytes",
    "canonical_endpoint_id",
    "decode_row_ordinal",
    "encode_row_ordinal",
    "endpoint_row_id",
    "iter_chunk_ranges",
    "iter_verified_parquet_rows",
    "iter_verified_rows",
    "logical_row_count",
    "logical_row_id",
    "logical_row_key",
    "ordered_numeric_aggregate",
    "semantic_row_bytes",
    "validate_streaming_schema_artifact",
    "verify_stream_artifact",
    "verify_parquet_stream_artifact",
    "write_parquet_stream_artifact",
    "write_stream_artifact",
]
