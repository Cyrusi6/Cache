from __future__ import annotations

import copy
import ast
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from script.analysis.fpct_e1_streaming_verify import (
    LOGICAL_ROW_ID_DOMAIN,
    MANIFEST_NAME,
    PARQUET_MANIFEST_NAME,
    PHYSICAL_CHUNK_ROWS,
    ROW_KEY_COLUMNS,
    SampleRowStream,
    attest_ordered_sample_streams,
    attest_ordinal_payload_stream,
    canonical_endpoint_id,
    canonical_json_bytes,
    decode_row_ordinal,
    encode_row_ordinal,
    endpoint_row_id,
    iter_chunk_ranges,
    iter_verified_parquet_rows,
    iter_verified_rows,
    logical_row_count,
    logical_row_id,
    logical_row_key,
    ordered_numeric_aggregate,
    semantic_row_bytes,
    validate_streaming_schema_artifact,
    verify_stream_artifact,
    verify_parquet_stream_artifact,
    write_parquet_stream_artifact,
    write_stream_artifact,
)
from script.experiment.fpct_e1_prepare_input_lock import row_template_records


SCHEMA_SHA256 = "a" * 64
ENDPOINT_ID = canonical_endpoint_id(
    45,
    "c_post",
    "f",
    "Y_CF",
    "ai2-arc",
    1.0,
)


def _item(query_count: int = 3, parent_count: int = 4) -> dict:
    parents = []
    for parent_index in range(parent_count):
        count = 2 + (parent_index % 3)
        indices = [parent_index * 10 + value for value in range(count)]
        weights = [1.0 / count] * count
        parents.append(
            {
                "parent_position": 10 + parent_index * 3,
                "candidate_count": count,
                "prior": weights,
                "candidate_indices": indices + [-1] * (4 - count),
                "candidate_valid_mask": [True] * count + [False] * (4 - count),
                "candidate_slot_weights": weights + [0.0] * (4 - count),
                "topology": (
                    "partition_compositional"
                    if parent_index % 2 == 0
                    else "competing_overlap"
                ),
                "statistical_weight": 1.0 / (parent_index + 1),
            }
        )
    return {
        "sample_sha256": "1" * 64,
        "content_group_sha256": "2" * 64,
        "provenance": {
            "input_sha256": "3" * 64,
            "alignment_sha256": "4" * 64,
            "labels_sha256": "5" * 64,
            "gold_response_sha256": "6" * 64,
        },
        "answer_queries": [
            {
                "query_position": 100 + query_index,
                "target_position": 101 + query_index,
                "target_token_id": 1000 + query_index,
            }
            for query_index in range(query_count)
        ],
        "certified_parents": parents,
    }


def _stream(
    *,
    query_count: int = 3,
    parent_count: int = 4,
    layers: int = 2,
    query_heads: int = 4,
    kv_heads: int = 2,
    endpoint_id: str = ENDPOINT_ID,
) -> SampleRowStream:
    return SampleRowStream(
        _item(query_count, parent_count),
        num_layers=layers,
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        endpoint_id=endpoint_id,
        schema_sha256=SCHEMA_SHA256,
        extra_row_factory=lambda ordinal, _row: {
            "metric_value": float((ordinal % 17) - 8) / 7.0,
            "metric_weight": float((ordinal % 5) + 1),
        },
    )


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _materialized_full_reference(
    item: dict,
    *,
    layers: int,
    query_heads: int,
    kv_heads: int,
) -> list[dict]:
    """Small synthetic-only oracle independent of SampleRowStream.row_at()."""

    rows = []
    group = query_heads // kv_heads
    ordinal = 0
    for layer in range(layers):
        for query_head in range(query_heads):
            for query in item["answer_queries"]:
                for parent in item["certified_parents"]:
                    provenance = item["provenance"]
                    row = {
                        "sample_sha256": item["sample_sha256"],
                        "content_group_sha256": item["content_group_sha256"],
                        "input_sha256": provenance["input_sha256"],
                        "alignment_sha256": provenance["alignment_sha256"],
                        "labels_sha256": provenance["labels_sha256"],
                        "gold_response_sha256": provenance["gold_response_sha256"],
                        "layer": layer,
                        "query_head": query_head,
                        "kv_head": query_head // group,
                        "query_position": query["query_position"],
                        "target_position": query["target_position"],
                        "target_token_id": query["target_token_id"],
                        "parent_position": parent["parent_position"],
                        "candidate_count": parent["candidate_count"],
                        "topology": parent["topology"],
                        "prior": list(parent["prior"]),
                        "candidate_indices": list(parent["candidate_indices"]),
                        "candidate_valid_mask": list(parent["candidate_valid_mask"]),
                        "candidate_slot_weights": list(parent["candidate_slot_weights"]),
                        "statistical_weight": parent["statistical_weight"],
                        "row_ordinal": ordinal,
                        "endpoint_id": ENDPOINT_ID,
                        "schema_sha256": SCHEMA_SHA256,
                        "metric_value": float((ordinal % 17) - 8) / 7.0,
                        "metric_weight": float((ordinal % 5) + 1),
                    }
                    key = [row[name] for name in ROW_KEY_COLUMNS]
                    row["logical_row_id"] = hashlib.sha256(
                        b"fpct-e1-a4-logical-row-id-v1\0"
                        + json.dumps(
                            key,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                            allow_nan=False,
                        ).encode("utf-8")
                    ).hexdigest()
                    row["endpoint_row_id"] = hashlib.sha256(
                        ENDPOINT_ID.encode("utf-8")
                        + b"\0"
                        + row["logical_row_id"].encode("ascii")
                    ).hexdigest()
                    rows.append(row)
                    ordinal += 1
    return rows


def _write_json(path: Path, value: dict) -> None:
    with path.open("wb") as handle:
        handle.write(canonical_json_bytes(value))


def test_ordinal_mapping_is_exhaustive_and_reversible() -> None:
    geometry = {
        "num_layers": 28,
        "num_query_heads": 16,
        "answer_query_count": 11,
        "certified_parent_count": 7,
    }
    count = logical_row_count(**geometry)
    assert count == 28 * 16 * 11 * 7
    observed = 0
    for layer in range(28):
        for head in range(16):
            for query in range(11):
                for parent in range(7):
                    ordinal = encode_row_ordinal(
                        layer=layer,
                        query_head=head,
                        query_index=query,
                        parent_index=parent,
                        **geometry,
                    )
                    assert ordinal == observed
                    assert decode_row_ordinal(ordinal, **geometry) == (
                        layer,
                        head,
                        query,
                        parent,
                    )
                    observed += 1
    assert observed == count
    with pytest.raises(ValueError, match="outside"):
        decode_row_ordinal(count, **geometry)


def test_logical_and_endpoint_id_contract_has_an_independent_byte_oracle() -> None:
    stream = _stream()
    row = stream.row_at(5)
    key = [row[name] for name in ROW_KEY_COLUMNS]
    key_bytes = json.dumps(
        key,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    expected_logical = hashlib.sha256(
        b"fpct-e1-a4-logical-row-id-v1\0" + key_bytes
    ).hexdigest()
    expected_endpoint = hashlib.sha256(
        ENDPOINT_ID.encode("utf-8")
        + b"\0"
        + expected_logical.encode("ascii")
    ).hexdigest()
    assert LOGICAL_ROW_ID_DOMAIN == b"fpct-e1-a4-logical-row-id-v1"
    assert logical_row_key(row) == tuple(key)
    assert logical_row_id(row) == expected_logical == row["logical_row_id"]
    assert endpoint_row_id(ENDPOINT_ID, expected_logical) == expected_endpoint
    assert row["endpoint_row_id"] == expected_endpoint
    other_endpoint = canonical_endpoint_id(
        46, "c_post", "f", "Y_CF", "ai2-arc", 1.0
    )
    other = _stream(endpoint_id=other_endpoint).row_at(5)
    assert other["logical_row_id"] == row["logical_row_id"]
    assert other["endpoint_row_id"] != row["endpoint_row_id"]


@pytest.mark.parametrize("query_count", [1, 2, 3])
@pytest.mark.parametrize("parent_count", [1, 2, 4])
@pytest.mark.parametrize("layers", [1, 2, 28])
@pytest.mark.parametrize("gqa_ratio", [1, 2, 4])
def test_streaming_row_key_reference_set_equivalence(
    query_count: int, parent_count: int, layers: int, gqa_ratio: int
) -> None:
    query_heads = 4
    kv_heads = query_heads // gqa_ratio
    item = _item(query_count, parent_count)
    stream = SampleRowStream(
        item,
        num_layers=layers,
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        endpoint_id=ENDPOINT_ID,
        schema_sha256=SCHEMA_SHA256,
        extra_row_factory=lambda ordinal, _row: {
            "metric_value": float((ordinal % 17) - 8) / 7.0,
            "metric_weight": float((ordinal % 5) + 1),
        },
    )
    materialized_reference = row_template_records(
        item,
        num_layers=layers,
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
    )
    reference_set = {tuple(values) for values in materialized_reference}
    observed_rows = list(stream.iter_rows())
    reference_rows = _materialized_full_reference(
        item, layers=layers, query_heads=query_heads, kv_heads=kv_heads
    )
    observed_set = set()
    for ordinal, row in enumerate(observed_rows):
        assert row["row_ordinal"] == ordinal
        assert row["kv_head"] == row["query_head"] // gqa_ratio
        parent = next(
            parent
            for parent in item["certified_parents"]
            if parent["parent_position"] == row["parent_position"]
        )
        assert row["candidate_count"] == parent["candidate_count"]
        assert row["topology"] == parent["topology"]
        assert row["prior"] == parent["prior"]
        assert row["statistical_weight"] == parent["statistical_weight"]
        observed_set.add(logical_row_key(row))
    assert len(observed_set) == stream.row_count == len(materialized_reference)
    assert observed_set == reference_set
    assert observed_rows == reference_rows
    assert hashlib.sha256(
        b"".join(semantic_row_bytes(row) for row in observed_rows)
    ).hexdigest() == hashlib.sha256(
        b"".join(canonical_json_bytes(row) for row in reference_rows)
    ).hexdigest()
    assert sum(row["metric_value"] * row["statistical_weight"] for row in observed_rows) == sum(
        row["metric_value"] * row["statistical_weight"] for row in reference_rows
    )


def test_default_physical_partition_is_exact_contiguous_cover() -> None:
    total = 616448
    chunks = tuple(iter_chunk_ranges(total))
    assert PHYSICAL_CHUNK_ROWS == 4096
    assert chunks[0].first_row_ordinal == 0
    assert all(left.end_row_ordinal_exclusive == right.first_row_ordinal for left, right in zip(chunks, chunks[1:]))
    assert chunks[-1].end_row_ordinal_exclusive == total
    assert sum(chunk.row_count for chunk in chunks) == total


def test_tracked_schema_validates_blocked_receipt_and_rejects_extra_fields() -> None:
    receipt = {
        "schema_version": 6,
        "protocol_id": "fpct_e1_mechanism_audit_v6_representation_preserving_streaming",
        "artifact_type": "streaming_input_lock_blocked_receipt",
        "status": "A4_INPUT_LOCK_BLOCKED",
        "failed_checks": ["synthetic_failure"],
        "runtime_probe_created": False,
        "execution_plan_created": False,
        "configmap_created": False,
        "checkpoint_job_created": False,
        "e1_2_started": False,
        "e1_3_started": False,
        "e1_pilot_consumed": False,
    }
    validate_streaming_schema_artifact(receipt)
    with pytest.raises(ValueError, match="oneOf"):
        validate_streaming_schema_artifact({**receipt, "unexpected": True})


def test_multi_sample_semantic_attestation_requires_sample_sha_ordinal_order() -> None:
    first = _stream(query_count=1, parent_count=1, layers=1, query_heads=2, kv_heads=1)
    second_item = _item(1, 1)
    second_item["sample_sha256"] = "7" * 64
    second_item["content_group_sha256"] = "8" * 64
    second = SampleRowStream(
        second_item,
        num_layers=1,
        num_query_heads=2,
        num_kv_heads=1,
        endpoint_id=ENDPOINT_ID,
        schema_sha256=SCHEMA_SHA256,
    )
    observed = attest_ordered_sample_streams((first, second))
    digest = hashlib.sha256()
    for stream in (first, second):
        for row in stream.iter_rows():
            digest.update(semantic_row_bytes(row))
    assert observed == {
        "sample_count": 2,
        "logical_row_count": first.row_count + second.row_count,
        "first_sample_sha256": "1" * 64,
        "last_sample_sha256": "7" * 64,
        "endpoint_id": ENDPOINT_ID,
        "schema_sha256": SCHEMA_SHA256,
        "semantic_stream_sha256": digest.hexdigest(),
    }
    with pytest.raises(ValueError, match="SHA order"):
        attest_ordered_sample_streams((second, first))


def test_endpoint_id_is_exact_six_field_compact_json_without_lf() -> None:
    expected = '[45,"c_post","f","Y_CF","ai2-arc",1.0]'
    assert ENDPOINT_ID == expected
    assert "\n" not in ENDPOINT_ID
    with pytest.raises(ValueError, match="six-field"):
        _stream(endpoint_id='[45,"c_post"]')
    with pytest.raises(ValueError, match="compact canonical"):
        _stream(endpoint_id='[45, "c_post", "f", "Y_CF", "ai2-arc", 1.0]')
    task_bound = _item(1, 1)
    task_bound["task"] = "openbookqa"
    with pytest.raises(ValueError, match="sample task"):
        SampleRowStream(
            task_bound,
            num_layers=1,
            num_query_heads=1,
            num_kv_heads=1,
            endpoint_id=ENDPOINT_ID,
            schema_sha256=SCHEMA_SHA256,
        )


def test_chunk_partition_does_not_change_semantics_ids_metrics_or_aggregate(
    tmp_path: Path,
) -> None:
    stream = _stream(layers=6, query_heads=4, kv_heads=1)
    baseline: tuple[str, list[str], list[tuple[float, float]], dict] | None = None
    physical_layouts = []
    for chunk_rows in (1, 7, 257, 4096, 8192):
        root = tmp_path / f"rows-{chunk_rows}"
        result = write_stream_artifact(root, stream, chunk_rows=chunk_rows)
        direct = attest_ordinal_payload_stream(
            (
                (row["row_ordinal"], semantic_row_bytes(row))
                for row in stream.iter_rows()
            ),
            expected_rows=stream.row_count,
        )
        assert direct["semantic_stream_sha256"] == result["semantic_stream_sha256"]
        rows = iter_verified_rows(root / MANIFEST_NAME, stream)
        ids = []
        metrics = []
        for row in rows:
            ids.append(row["logical_row_id"])
            metrics.append((row["metric_value"], row["metric_weight"]))
        aggregate = ordered_numeric_aggregate(
            root / MANIFEST_NAME, stream, ("metric_value", "metric_weight")
        )
        current = (result["semantic_stream_sha256"], ids, metrics, aggregate)
        if baseline is None:
            baseline = current
        else:
            assert current == baseline
        manifest = _read_json(root / MANIFEST_NAME)
        physical_layouts.append(
            tuple((row["row_count"], row["physical_file_sha256"]) for row in manifest["chunks"])
        )
    assert baseline is not None
    assert len(set(physical_layouts)) > 1


def test_parquet_writer_uses_v6_schema_and_is_partition_invariant(
    tmp_path: Path,
) -> None:
    stream = _stream(query_count=2, parent_count=2, layers=2, query_heads=2, kv_heads=1)
    baseline: tuple[str, list[str], list[tuple[float, float]]] | None = None
    layouts = []
    for chunk_rows in (1, 7, 257, 4096, 8192):
        root = tmp_path / f"parquet-{chunk_rows}"
        result = write_parquet_stream_artifact(
            root,
            stream,
            chunk_rows=chunk_rows,
            synthetic_partition_test=chunk_rows != 4096,
        )
        rows = iter_verified_parquet_rows(root / PARQUET_MANIFEST_NAME, stream)
        ids = []
        metrics = []
        for row in rows:
            ids.append(row["logical_row_id"])
            metrics.append((row["metric_value"], row["metric_weight"]))
        current = (result["semantic_stream_sha256"], ids, metrics)
        if baseline is None:
            baseline = current
        else:
            assert current == baseline
        manifest = _read_json(root / PARQUET_MANIFEST_NAME)
        layouts.append(
            tuple(
                (record["row_count"], record["physical_file_sha256"])
                for record in manifest["chunks"]
            )
        )
        assert manifest["schema_version"] == 6
        assert manifest["protocol_id"] == (
            "fpct_e1_mechanism_audit_v6_representation_preserving_streaming"
        )
        if chunk_rows == 4096:
            schema = _read_json(
                Path(__file__).resolve().parents[1]
                / "recipe/eval_recipe/fpct_e1/e1_streaming_schema.json"
            )
            contract = schema["$defs"]["chunkManifest"]
            assert set(manifest) == set(contract["required"])
            assert manifest["physical_chunk_rows"] == contract["properties"][
                "physical_chunk_rows"
            ]["const"]
        if chunk_rows == 8192:
            assert manifest["synthetic_logical_partition_rows"] == 8192
            assert manifest["physical_chunk_rows"] == 4096
            assert max(record["row_count"] for record in manifest["chunks"]) <= 4096
    assert baseline is not None
    assert len(set(layouts)) > 1
    direct = attest_ordinal_payload_stream(
        (
            (row["row_ordinal"], semantic_row_bytes(row))
            for row in stream.iter_rows()
        ),
        expected_rows=stream.row_count,
    )
    assert baseline[0] == direct["semantic_stream_sha256"]
    with pytest.raises(ValueError, match="must equal 4096"):
        write_parquet_stream_artifact(
            tmp_path / "non-production", stream, chunk_rows=7
        )


@pytest.mark.parametrize("alias_kind", ("root", "chunk", "manifest"))
def test_parquet_producer_preflight_rejects_alias_with_zero_external_writes(
    tmp_path: Path, alias_kind: str
) -> None:
    stream = _stream(
        query_count=2, parent_count=2, layers=2, query_heads=2, kv_heads=1
    )
    root = tmp_path / "hostile-parquet"
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "marker.bin"
    marker.write_bytes(b"immutable external bytes")
    if alias_kind == "root":
        root.symlink_to(external, target_is_directory=True)
    else:
        root.mkdir()
        name = (
            "chunk_00000000.parquet"
            if alias_kind == "chunk"
            else PARQUET_MANIFEST_NAME
        )
        (root / name).symlink_to(marker)
    before = marker.read_bytes()
    with pytest.raises(ValueError, match="canonical|regular"):
        write_parquet_stream_artifact(root, stream)
    assert marker.read_bytes() == before
    assert sorted(path.name for path in external.iterdir()) == ["marker.bin"]


def test_parquet_verifier_rejects_manifest_and_physical_corruption(
    tmp_path: Path,
) -> None:
    import pyarrow

    stream = _stream(query_count=1, parent_count=2, layers=1, query_heads=2, kv_heads=1)
    manifest_root = tmp_path / "manifest-corruption"
    write_parquet_stream_artifact(manifest_root, stream)
    manifest_path = manifest_root / PARQUET_MANIFEST_NAME
    manifest = _read_json(manifest_path)
    manifest["chunks"][0]["physical_file_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="manifest SHA"):
        verify_parquet_stream_artifact(manifest_path, stream)

    file_root = tmp_path / "file-corruption"
    write_parquet_stream_artifact(file_root, stream)
    file_manifest = _read_json(file_root / PARQUET_MANIFEST_NAME)
    parquet_path = file_root / file_manifest["chunks"][0]["relative_path"]
    with parquet_path.open("ab") as handle:
        handle.write(b"not-a-parquet-footer")
    with pytest.raises((ValueError, OSError, pyarrow.ArrowInvalid)):
        verify_parquet_stream_artifact(
            file_root / PARQUET_MANIFEST_NAME, stream
        )

    orphan_root = tmp_path / "orphan-corruption"
    write_parquet_stream_artifact(orphan_root, stream)
    (orphan_root / "chunk_99999999.parquet").write_bytes(b"orphan")
    with pytest.raises(ValueError, match="orphan"):
        verify_parquet_stream_artifact(
            orphan_root / PARQUET_MANIFEST_NAME, stream
        )


@pytest.mark.parametrize(
    "corruption",
    (
        "missing_chunk",
        "duplicate_chunk",
        "overlap",
        "gap",
        "wrong_sample",
        "wrong_schema",
        "wrong_expected_n",
        "wrong_chunk_sha",
        "wrong_semantic_sha",
        "row_mutation",
        "row_reorder",
        "wrong_endpoint",
    ),
)
def test_production_parquet_verifier_rejects_full_corruption_matrix(
    tmp_path: Path, corruption: str
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    stream = _stream(query_count=2, parent_count=2, layers=2, query_heads=2, kv_heads=1)
    root = tmp_path / corruption
    write_parquet_stream_artifact(
        root, stream, chunk_rows=7, synthetic_partition_test=True
    )
    manifest_path = root / PARQUET_MANIFEST_NAME
    manifest = _read_json(manifest_path)
    if corruption == "missing_chunk":
        manifest["chunks"].pop()
    elif corruption == "duplicate_chunk":
        manifest["chunks"].append(copy.deepcopy(manifest["chunks"][-1]))
    elif corruption == "overlap":
        manifest["chunks"][1]["first_row_ordinal"] -= 1
        manifest["chunks"][1]["row_count"] += 1
    elif corruption == "gap":
        manifest["chunks"][1]["first_row_ordinal"] += 1
        manifest["chunks"][1]["row_count"] -= 1
    elif corruption == "wrong_sample":
        manifest["sample_sha256"] = "9" * 64
    elif corruption == "wrong_schema":
        manifest["chunks"][0]["schema_sha256"] = "b" * 64
    elif corruption == "wrong_expected_n":
        manifest["expected_logical_rows"] += 1
    elif corruption == "wrong_chunk_sha":
        manifest["chunks"][0]["physical_file_sha256"] = "0" * 64
    elif corruption == "wrong_semantic_sha":
        manifest["semantic_stream_sha256"] = "0" * 64
    else:
        record = manifest["chunks"][0]
        path = root / record["relative_path"]
        table = pq.read_table(path)
        rows = table.to_pylist()
        if corruption == "row_mutation":
            rows[0]["metric_value"] += 1.0
        elif corruption == "row_reorder":
            rows[0], rows[1] = rows[1], rows[0]
        elif corruption == "wrong_endpoint":
            rows[0]["endpoint_id"] = canonical_endpoint_id(
                46, "c_post", "f", "Y_CF", "ai2-arc", 1.0
            )
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        record["physical_file_bytes"] = path.stat().st_size
        record["physical_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        verify_parquet_stream_artifact(manifest_path, stream)


@pytest.mark.parametrize(
    "stage",
    [
        "chunk_temporary_half_written",
        "before_chunk_close",
        "before_artifact_rename",
        "before_manifest_write",
        "before_manifest_rename",
    ],
)
def test_parquet_crash_resume_reuses_completed_chunks_and_replays_semantics(
    tmp_path: Path, stage: str
) -> None:
    stream = _stream(query_count=2, parent_count=2, layers=2, query_heads=2, kv_heads=1)
    clean_root = tmp_path / "clean-parquet"
    clean = write_parquet_stream_artifact(
        clean_root, stream, chunk_rows=4, synthetic_partition_test=True
    )
    crash_root = tmp_path / "crash-parquet"
    fired = False

    def fail_at(candidate_stage: str, chunk_index: int | None) -> None:
        nonlocal fired
        target_chunk = None if stage.startswith("before_manifest") else 1
        if not fired and candidate_stage == stage and chunk_index == target_chunk:
            fired = True
            raise RuntimeError(f"synthetic parquet crash at {stage}")

    with pytest.raises(RuntimeError, match="synthetic parquet crash"):
        write_parquet_stream_artifact(
            crash_root,
            stream,
            chunk_rows=4,
            synthetic_partition_test=True,
            failure_injector=fail_at,
        )
    assert fired
    assert not (crash_root / PARQUET_MANIFEST_NAME).exists()
    completed = crash_root / "chunk_00000000.parquet"
    completed_stat = completed.stat()
    completed_sha = hashlib.sha256(completed.read_bytes()).hexdigest()
    resumed = write_parquet_stream_artifact(
        crash_root, stream, chunk_rows=4, synthetic_partition_test=True
    )
    assert resumed["semantic_stream_sha256"] == clean["semantic_stream_sha256"]
    assert completed.stat().st_ino == completed_stat.st_ino
    assert completed.stat().st_mtime_ns == completed_stat.st_mtime_ns
    assert hashlib.sha256(completed.read_bytes()).hexdigest() == completed_sha
    assert not tuple(crash_root.glob(".*.tmp"))


def test_parquet_verification_has_no_whole_table_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow.parquet as parquet
    import pandas

    stream = _stream(query_count=2, parent_count=2, layers=2, query_heads=2, kv_heads=1)
    root = tmp_path / "parquet-no-whole-table"
    write_parquet_stream_artifact(root, stream)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("whole-table parquet entrypoint was called")

    monkeypatch.setattr(parquet, "read_table", forbidden)
    monkeypatch.setattr(pandas, "read_parquet", forbidden)
    result = verify_parquet_stream_artifact(
        root / PARQUET_MANIFEST_NAME, stream
    )
    assert result["status"] == "GO"
    source = (
        Path(__file__).resolve().parents[1]
        / "script/analysis/fpct_e1_streaming_verify.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"read_table", "to_pylist", "read_parquet"}
    }
    assert forbidden_calls == set()


def test_million_row_stress_is_repeatable_contiguous_and_bounded_rss() -> None:
    # Run in a fresh interpreter so ru_maxrss is a meaningful absolute bound,
    # rather than a pytest-process historical high-water mark.
    code = """
import json, resource
from script.analysis.fpct_e1_streaming_verify import attest_ordinal_payload_stream
n = int(__import__('sys').argv[1])
def run():
    rows = ((i, b'{"row_ordinal":' + str(i).encode('ascii') + b'}\\n') for i in range(n))
    return attest_ordinal_payload_stream(rows, expected_rows=n)
first = run()
second = run()
print(json.dumps({'first': first, 'second': second, 'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}, sort_keys=True))
"""
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    def invoke(row_count: int) -> dict:
        completed = subprocess.run(
            [sys.executable, "-c", code, str(row_count)],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    baseline = invoke(10_003)
    payload = invoke(1_000_003)
    assert payload["first"] == payload["second"]
    assert payload["first"]["count"] == 1_000_003
    assert payload["first"]["first_row_ordinal"] == 0
    assert payload["first"]["last_row_ordinal"] == 1_000_002
    assert payload["first"]["physical_chunk_rows"] == 4096
    assert payload["first"]["max_rows_in_chunk"] == 4096
    assert payload["first"]["chunk_count"] == 245
    # Linux reports KiB.  This scaling check proves that increasing logical N by
    # ~100x does not allocate an N-sized table.  It is not the operative A4 RSS
    # threshold: the protocol requires that separately derived immutable
    # synthetic RSS lock before any successor natural input is accessed.
    assert payload["maxrss_kib"] <= baseline["maxrss_kib"] + 16 * 1024


def _corrupt_manifest_or_file(root: Path, corruption: str) -> None:
    manifest_path = root / MANIFEST_NAME
    manifest = _read_json(manifest_path)
    if corruption == "missing_chunk":
        manifest["chunks"].pop()
    elif corruption == "duplicate_chunk":
        manifest["chunks"].append(copy.deepcopy(manifest["chunks"][-1]))
    elif corruption == "overlapping_range":
        manifest["chunks"][1]["first_row_ordinal"] -= 1
        manifest["chunks"][1]["row_count"] += 1
    elif corruption == "ordinal_gap":
        manifest["chunks"][1]["first_row_ordinal"] += 1
        manifest["chunks"][1]["row_count"] -= 1
    elif corruption == "wrong_sample_hash":
        manifest["sample_sha256"] = "9" * 64
    elif corruption == "wrong_endpoint":
        manifest["endpoint_id"] = "wrong-endpoint"
    elif corruption == "wrong_schema":
        manifest["schema_sha256"] = "b" * 64
    elif corruption == "wrong_expected_n":
        manifest["expected_logical_rows"] += 1
    elif corruption == "wrong_chunk_sha":
        manifest["chunks"][0]["physical_file_sha256"] = "0" * 64
    elif corruption == "wrong_semantic_sha":
        manifest["semantic_stream_sha256"] = "0" * 64
    elif corruption in {"row_mutation", "row_reordered"}:
        chunk_path = root / manifest["chunks"][0]["physical_file"]
        with chunk_path.open("rb") as handle:
            lines = handle.readlines()
        if corruption == "row_mutation":
            row = json.loads(lines[0])
            row["metric_value"] += 1.0
            lines[0] = canonical_json_bytes(row)
        else:
            lines[0], lines[1] = lines[1], lines[0]
        with chunk_path.open("wb") as handle:
            handle.writelines(lines)
        return
    else:  # pragma: no cover - test authoring error
        raise AssertionError(corruption)
    _write_json(manifest_path, manifest)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_chunk",
        "duplicate_chunk",
        "overlapping_range",
        "ordinal_gap",
        "wrong_sample_hash",
        "wrong_endpoint",
        "wrong_schema",
        "wrong_expected_n",
        "wrong_chunk_sha",
        "wrong_semantic_sha",
        "row_mutation",
        "row_reordered",
    ],
)
def test_verifier_rejects_every_frozen_corruption(
    tmp_path: Path, corruption: str
) -> None:
    stream = _stream(query_count=2, parent_count=2, layers=2, query_heads=2, kv_heads=1)
    root = tmp_path / corruption
    write_stream_artifact(root, stream, chunk_rows=4)
    _corrupt_manifest_or_file(root, corruption)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_stream_artifact(root / MANIFEST_NAME, stream)


@pytest.mark.parametrize(
    "stage",
    [
        "chunk_temporary_half_written",
        "before_chunk_close",
        "before_artifact_rename",
        "before_manifest_write",
        "before_manifest_rename",
    ],
)
def test_crash_resume_is_atomic_idempotent_and_semantically_exact(
    tmp_path: Path, stage: str
) -> None:
    stream = _stream(query_count=2, parent_count=2, layers=2, query_heads=2, kv_heads=1)
    clean_root = tmp_path / "clean"
    clean = write_stream_artifact(clean_root, stream, chunk_rows=4)
    crash_root = tmp_path / "crash"
    fired = False

    def fail_at(candidate_stage: str, chunk_index: int | None) -> None:
        nonlocal fired
        target_chunk = None if stage.startswith("before_manifest") else 1
        if not fired and candidate_stage == stage and chunk_index == target_chunk:
            fired = True
            raise RuntimeError(f"synthetic crash at {stage}")

    with pytest.raises(RuntimeError, match="synthetic crash"):
        write_stream_artifact(
            crash_root,
            stream,
            chunk_rows=4,
            failure_injector=fail_at,
        )
    assert fired
    assert not (crash_root / MANIFEST_NAME).exists()
    completed = crash_root / "chunk_00000000.jsonl"
    assert completed.is_file()
    completed_stat = completed.stat()
    completed_sha = hashlib.sha256(completed.read_bytes()).hexdigest()
    temporary_files = tuple(crash_root.glob("*.tmp")) + tuple(crash_root.glob(".*.tmp"))
    if stage != "before_manifest_write":
        assert temporary_files

    resumed = write_stream_artifact(crash_root, stream, chunk_rows=4)
    assert resumed["semantic_stream_sha256"] == clean["semantic_stream_sha256"]
    assert resumed["emitted_logical_rows"] == stream.row_count
    assert completed.stat().st_ino == completed_stat.st_ino
    assert completed.stat().st_mtime_ns == completed_stat.st_mtime_ns
    assert hashlib.sha256(completed.read_bytes()).hexdigest() == completed_sha
    assert not tuple(crash_root.glob("*.tmp"))
    assert not tuple(crash_root.glob(".*.tmp"))


def test_completed_chunks_are_immutable_and_corruption_is_not_overwritten(
    tmp_path: Path,
) -> None:
    stream = _stream(query_count=1, parent_count=2, layers=1, query_heads=2, kv_heads=1)
    root = tmp_path / "immutable"
    write_stream_artifact(root, stream, chunk_rows=2)
    (root / MANIFEST_NAME).unlink()
    chunk = root / "chunk_00000000.jsonl"
    with chunk.open("ab") as handle:
        handle.write(b"{}\n")
    with pytest.raises(ValueError):
        write_stream_artifact(root, stream, chunk_rows=2)


def test_streaming_path_never_uses_whole_table_entrypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = _stream(query_count=2, parent_count=2, layers=2, query_heads=2, kv_heads=1)
    root = tmp_path / "no-whole-table"
    write_stream_artifact(root, stream, chunk_rows=3)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("whole-table materialization entrypoint was called")

    fake_parquet = types.ModuleType("pyarrow.parquet")
    fake_parquet.read_table = forbidden  # type: ignore[attr-defined]
    fake_arrow = types.ModuleType("pyarrow")
    fake_arrow.parquet = fake_parquet  # type: ignore[attr-defined]
    fake_pandas = types.ModuleType("pandas")
    fake_pandas.read_parquet = forbidden  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyarrow", fake_arrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.suffix == ".jsonl":
            raise AssertionError("large JSONL was read as a whole")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    result = verify_stream_artifact(root / MANIFEST_NAME, stream)
    assert result["status"] == "GO"
    assert sum(1 for _row in iter_verified_rows(root / MANIFEST_NAME, stream)) == stream.row_count


def test_validation_rejects_nonfinite_metrics_and_protected_overwrite() -> None:
    item = _item(1, 1)
    invalid = SampleRowStream(
        item,
        num_layers=1,
        num_query_heads=1,
        num_kv_heads=1,
        endpoint_id=ENDPOINT_ID,
        schema_sha256=SCHEMA_SHA256,
        extra_row_factory=lambda _ordinal, _row: {"metric": float("nan")},
    )
    with pytest.raises(ValueError):
        invalid.row_at(0)
    overwrite = SampleRowStream(
        item,
        num_layers=1,
        num_query_heads=1,
        num_kv_heads=1,
        endpoint_id=ENDPOINT_ID,
        schema_sha256=SCHEMA_SHA256,
        extra_row_factory=lambda _ordinal, _row: {"row_ordinal": 99},
    )
    with pytest.raises(ValueError, match="protected"):
        overwrite.row_at(0)


def test_payload_attestation_rejects_gap_and_short_stream() -> None:
    with pytest.raises(ValueError, match="gap"):
        attest_ordinal_payload_stream(((0, b"a\n"), (2, b"b\n")), expected_rows=2)
    with pytest.raises(ValueError, match="count differs"):
        attest_ordinal_payload_stream(((0, b"a\n"),), expected_rows=2)
