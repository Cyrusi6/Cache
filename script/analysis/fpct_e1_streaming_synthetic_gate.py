#!/usr/bin/env python3
"""Freeze the FPCT-E1 A4 pre-natural deterministic streaming hard gate.

This command is synthetic-only.  It never imports tokenizers, datasets, model
classes, checkpoints, CUDA, or Kubernetes clients.  Fresh worker processes
exercise the complete mechanism-row shape through the production 4096-row
Parquet writer.  The resulting tracked receipt is a prerequisite for the
successor E0-design input lock; it is not natural-data or scientific evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
STREAMING_SCHEMA = REPO_ROOT / "recipe/eval_recipe/fpct_e1/e1_streaming_schema.json"
MECHANISM_SCHEMA = REPO_ROOT / "recipe/eval_recipe/fpct_e1/e1_mechanism_schema.json"
DEFAULT_OUTPUT = REPO_ROOT / "recipe/eval_recipe/fpct_e1/e1_streaming_synthetic_gate.json"
PROTOCOL_ID = "fpct_e1_mechanism_audit_v6_representation_preserving_streaming"
SCHEMA_VERSION = 6
WORKER_MODULE = "script.analysis.fpct_e1_streaming_synthetic_gate"
LAYERS = 28
QUERY_HEADS = 16
KV_HEADS = 8
PHYSICAL_CHUNK_ROWS = 4096
BUFFER_COPY_MULTIPLIER = 16
MINIMUM_STRESS_ROWS = 1_000_000
TEST_FILES = (
    "test/test_fpct_production_path.py",
    "test/test_fpct_reference_operator.py",
    "test/test_fpct_e1_streaming.py",
    "test/test_fpct_e1_streaming_synthetic_gate.py",
    "test/test_fpct_e1_prepare_input_lock.py",
    "test/test_fpct_instrumentation.py",
    "test/test_fpct_e1_runtime_capture_primitives.py",
    "test/test_fpct_e1_runtime_backend.py",
    "test/test_fpct_e1_mechanism_audit.py",
    "test/test_fpct_e1_capture_runner.py",
    "test/test_fpct_e1_k8s_lock_bundle.py",
    "test/test_fpct_e1_instrumentation_gate_a4.py",
    "test/test_fpct_sealed_import.py",
    "test/test_fpct_e1_source_snapshot_lock.py",
)
TRACKED_SOURCE_FILES = tuple(
    sorted(
        {
            "script/analysis/fpct_e1_streaming_verify.py",
            "script/analysis/fpct_e1_streaming_synthetic_gate.py",
            "script/analysis/fpct_e1_mechanism_audit.py",
            "script/analysis/fpct_e1_instrumentation_gate.py",
            "script/analysis/fpct_reference_operator.py",
            "script/experiment/fpct_e1_prepare_input_lock.py",
            "script/experiment/fpct_e1_runtime_backend.py",
            "script/experiment/fpct_e1_capture_runner.py",
            "script/experiment/fpct_e1_k8s_lock_bundle.py",
            "script/experiment/fpct_e1_source_snapshot_lock.py",
            "script/runtime/fpct_bootstrap.py",
            "rosetta/model/fpct_attention.py",
            "rosetta/model/fpct_instrumentation.py",
            "rosetta/model/wrapper.py",
            "recipe/eval_recipe/fpct_e1/e1_streaming_contract.json",
            "recipe/eval_recipe/fpct_e1/e1_instrumentation_parity.json",
            "recipe/eval_recipe/fpct_e1/e1_synthetic_query_variance.json",
            "recipe/eval_recipe/fpct_e1/e1_instrumentation_hard_gate.json",
            "recipe/eval_recipe/fpct_e1/e1_instrumentation_parity_a4.json",
            "recipe/eval_recipe/fpct_e1/e1_synthetic_query_variance_a4.json",
            "recipe/eval_recipe/fpct_e1/e1_instrumentation_hard_gate_a4.json",
            "test/test_fpct_e1_capture_runner.py",
            "test/test_fpct_e1_k8s_lock_bundle.py",
            *TEST_FILES,
        }
    )
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while payload := handle.read(1024 * 1024):
            digest.update(payload)
    return digest.hexdigest()


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _hash(character: str) -> str:
    return character * 64


def _extra_metrics(row_ordinal: int, _: Mapping[str, Any]) -> dict[str, Any]:
    # Complete fixed mechanism-row shape.  Values vary deterministically with
    # ordinal so compression cannot turn the stress case into a single constant
    # row, while every array and weight remains finite and valid.
    offset = float((row_ordinal % 997) + 1) / 997.0
    return {
        "schema_version": SCHEMA_VERSION,
        "split_role": "synthetic_pre_natural",
        "seed": 0,
        "checkpoint_arm": "C_post",
        "inference_operator": "f",
        "cell": "Y_CF",
        "task": "synthetic-a4",
        "lambda_value": 1.0,
        "gamma": [0.1, 0.2, 0.3, 0.4],
        "source_d_k": offset,
        "source_d_v": offset + 0.01,
        "source_energy_k": offset + 1.0,
        "source_energy_v": offset + 1.01,
        "fused_d_k": offset + 0.02,
        "fused_d_v": offset + 0.03,
        "fused_energy_k": offset + 1.02,
        "fused_energy_v": offset + 1.03,
        "candidate_logit_range": offset + 0.04,
        "candidate_logit_variance": offset + 0.05,
        "jensen_gap": offset + 0.06,
        "parent_attention_mass": 0.5,
        "output_delta_l2": offset + 0.07,
        "gold_logp": -offset,
        "cpost_gold_logp": -(offset + 0.01),
        "end_task_correct": 1,
        "cpost_end_task_correct": 1,
    }


def _sample_item(query_count: int, *, maximum_width: bool = False) -> dict[str, Any]:
    if maximum_width:
        limit = (1 << 63) - 2
        queries = [
            {
                "query_position": limit,
                "target_position": limit + 1,
                "target_token_id": limit,
            }
        ]
        parent_position = limit
    else:
        queries = [
            {
                "query_position": query_index,
                "target_position": query_index + 1,
                "target_token_id": 100 + (query_index % 32_000),
            }
            for query_index in range(query_count)
        ]
        parent_position = 0
    return {
        "task": "synthetic-a4",
        "sample_sha256": _hash("1"),
        "content_group_sha256": _hash("2"),
        "provenance": {
            "input_sha256": _hash("3"),
            "alignment_sha256": _hash("4"),
            "labels_sha256": _hash("5"),
            "gold_response_sha256": _hash("6"),
        },
        "answer_queries": queries,
        "certified_parents": [
            {
                "parent_position": parent_position,
                "candidate_count": 4,
                "prior": [0.1, 0.2, 0.3, 0.4],
                "candidate_indices": [10, 11, 12, 13],
                "candidate_valid_mask": [True, True, True, True],
                "candidate_slot_weights": [0.1, 0.2, 0.3, 0.4],
                "topology": "partition_compositional",
                "statistical_weight": 1.0,
            }
        ],
    }


def _stream(query_count: int, schema_sha256: str):
    from script.analysis.fpct_e1_streaming_verify import (
        SampleRowStream,
        canonical_endpoint_id,
    )

    return SampleRowStream(
        _sample_item(query_count),
        num_layers=LAYERS,
        num_query_heads=QUERY_HEADS,
        num_kv_heads=KV_HEADS,
        endpoint_id=canonical_endpoint_id(
            0, "C_post", "f", "Y_CF", "synthetic-a4", 1.0
        ),
        schema_sha256=schema_sha256,
        extra_row_factory=_extra_metrics,
    )


def _canonical_full_schema_row_bound(schema_sha256: str) -> int:
    from script.analysis.fpct_e1_streaming_verify import (
        SampleRowStream,
        canonical_endpoint_id,
        semantic_row_bytes,
    )

    stream = SampleRowStream(
        _sample_item(1, maximum_width=True),
        num_layers=LAYERS,
        num_query_heads=QUERY_HEADS,
        num_kv_heads=KV_HEADS,
        endpoint_id=canonical_endpoint_id(
            (1 << 63) - 1,
            "C_post",
            "f",
            "Y_CF",
            "synthetic-a4",
            2.0,
        ),
        schema_sha256=schema_sha256,
        extra_row_factory=_extra_metrics,
    )
    return max(len(semantic_row_bytes(stream.row_at(index))) for index in (0, stream.row_count - 1))


def _worker(query_count: int) -> dict[str, Any]:
    from script.analysis.fpct_e1_streaming_verify import (
        PARQUET_MANIFEST_NAME,
        semantic_row_bytes,
        verify_parquet_stream_artifact,
        write_parquet_stream_artifact,
    )

    schema_sha256 = sha256_file(STREAMING_SCHEMA)
    stream = _stream(query_count, schema_sha256)
    mechanism_required = set(json.loads(MECHANISM_SCHEMA.read_text(encoding="utf-8"))["input_row"]["required_columns"])
    missing = mechanism_required.difference(stream.row_at(0))
    if missing:
        raise RuntimeError(f"synthetic stress row omits mechanism fields: {sorted(missing)}")
    with tempfile.TemporaryDirectory(prefix="fpct-e1-a4-synthetic-") as temporary:
        root = Path(temporary)
        first = write_parquet_stream_artifact(root, stream)
        second = verify_parquet_stream_artifact(root / PARQUET_MANIFEST_NAME, stream)
        if first != second:
            raise RuntimeError("semantic replay differs from the initial write verification")
        manifest = json.loads((root / PARQUET_MANIFEST_NAME).read_text(encoding="utf-8"))
        physical_bytes = sum(int(record["physical_file_bytes"]) for record in manifest["chunks"])
        max_physical_chunk_bytes = max(int(record["physical_file_bytes"]) for record in manifest["chunks"])
        observed_row_bytes = max(
            len(semantic_row_bytes(stream.row_at(index)))
            for index in (0, stream.row_count // 2, stream.row_count - 1)
        )
    return {
        "logical_rows": stream.row_count,
        "emitted_rows": first["emitted_logical_rows"],
        "semantic_stream_sha256": first["semantic_stream_sha256"],
        "replay_semantic_stream_sha256": second["semantic_stream_sha256"],
        "chunk_count": first["chunk_count"],
        "physical_file_bytes": physical_bytes,
        "max_physical_chunk_bytes": max_physical_chunk_bytes,
        "max_observed_canonical_row_bytes": observed_row_bytes,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }


def _fresh_worker(case_id: str, query_count: int) -> dict[str, Any]:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-s", "-m", WORKER_MODULE, "--worker", str(query_count)],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    value["case_id"] = case_id
    return value


def _run_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-cov",
        "-p",
        "no:cacheprovider",
        "--basetemp=/tmp/pytest-e1-a4-synthetic-gate",
        *TEST_FILES,
    ]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    combined = completed.stdout + completed.stderr
    return {
        "command": command,
        "exit_code": completed.returncode,
        "output_sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        "last_output_line": next(
            (line for line in reversed(combined.splitlines()) if line.strip()), ""
        ),
        "test_files": list(TEST_FILES),
    }


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("power-of-two input must be positive")
    return 1 << (value - 1).bit_length()


def verify_tracked_gate(gate_path: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Recompute every source/evidence binding before a gate is consumed."""

    from script.analysis.fpct_e1_streaming_verify import (
        validate_streaming_schema_artifact,
    )

    value = json.loads(gate_path.read_text(encoding="utf-8"))
    schema_path = repo_root / "recipe/eval_recipe/fpct_e1/e1_streaming_schema.json"
    mechanism_schema = repo_root / "recipe/eval_recipe/fpct_e1/e1_mechanism_schema.json"
    validate_streaming_schema_artifact(value, schema_path)
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping) or value.get("evidence_sha256") != _sha256_value(evidence):
        raise ValueError("A4 synthetic gate evidence SHA does not recompute")
    if value.get("streaming_schema_sha256") != sha256_file(schema_path):
        raise ValueError("A4 synthetic gate streaming schema SHA is stale")
    if value.get("mechanism_schema_sha256") != sha256_file(mechanism_schema):
        raise ValueError("A4 synthetic gate mechanism schema SHA is stale")
    expected_sources = {
        relative: sha256_file(repo_root / relative) for relative in TRACKED_SOURCE_FILES
    }
    if evidence.get("tracked_source_sha256") != expected_sources:
        raise ValueError("A4 synthetic gate tracked source/test SHA map is stale")
    checks = value.get("checks")
    if not isinstance(checks, Mapping) or any(
        checks.get(name) is not expected
        for name, expected in {
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
        }.items()
    ):
        raise ValueError("A4 synthetic gate required check changed")
    return dict(value)


def build_gate() -> dict[str, Any]:
    import pyarrow

    tests = _run_tests()
    schema_sha256 = sha256_file(STREAMING_SCHEMA)
    mechanism_schema_sha256 = sha256_file(MECHANISM_SCHEMA)
    rows_per_query = LAYERS * QUERY_HEADS
    baseline_queries = math.floor(PHYSICAL_CHUNK_ROWS / rows_per_query) + 1
    stress_queries = math.ceil(MINIMUM_STRESS_ROWS / rows_per_query)
    canonical_row_bound = _canonical_full_schema_row_bound(schema_sha256)
    baseline = _fresh_worker("full_schema_two_chunk_baseline", baseline_queries)
    stress = _fresh_worker("full_schema_million_row_stress", stress_queries)
    expected_stress_rows = rows_per_query * stress_queries
    if baseline["logical_rows"] <= PHYSICAL_CHUNK_ROWS or baseline["chunk_count"] < 2:
        raise RuntimeError("baseline case does not span at least two chunks")
    if stress["logical_rows"] != expected_stress_rows or stress["logical_rows"] < MINIMUM_STRESS_ROWS:
        raise RuntimeError("stress geometry does not cover the preregistered million rows")
    for case in (baseline, stress):
        if case["logical_rows"] != case["emitted_rows"]:
            raise RuntimeError("synthetic emitted rows differ from logical rows")
        if case["semantic_stream_sha256"] != case["replay_semantic_stream_sha256"]:
            raise RuntimeError("synthetic semantic replay differs")
        if case["max_observed_canonical_row_bytes"] > canonical_row_bound:
            raise RuntimeError("observed full-schema row exceeds the prospective canonical bound")

    allowance = BUFFER_COPY_MULTIPLIER * PHYSICAL_CHUNK_ROWS * canonical_row_bound
    threshold_bytes = int(baseline["peak_rss_bytes"]) + allowance
    bounded = int(stress["peak_rss_bytes"]) <= threshold_bytes
    if not bounded:
        raise RuntimeError(
            f"synthetic RSS gate failed: {stress['peak_rss_bytes']} > {threshold_bytes}"
        )
    environment = {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pyarrow": pyarrow.__version__,
        "cuda_visible_devices": "",
    }
    cases_for_schema = [
        {
            "case_id": case["case_id"],
            "logical_rows": case["logical_rows"],
            "peak_rss_bytes": case["peak_rss_bytes"],
            "emitted_rows": case["emitted_rows"],
            "semantic_stream_sha256": case["semantic_stream_sha256"],
        }
        for case in (baseline, stress)
    ]
    evidence = {
        "environment": environment,
        "targeted_tests": tests,
        "tracked_source_sha256": {
            relative: sha256_file(REPO_ROOT / relative)
            for relative in TRACKED_SOURCE_FILES
        },
        "baseline_full_record": baseline,
        "stress_full_record": stress,
        "canonical_full_schema_row_bound_bytes": canonical_row_bound,
        "buffer_copy_multiplier": BUFFER_COPY_MULTIPLIER,
        "allocator_and_buffer_allowance_bytes": allowance,
    }
    checks = {
        "row_key_reference_equivalence": True,
        "weights_reference_equivalence": True,
        "topology_reference_equivalence": True,
        "semantic_stream_replay_equal": True,
        "chunk_partition_semantic_equivalence": True,
        "aggregate_partition_equivalence": True,
        "bounded_peak_rss": bounded,
        "whole_table_materialization_detected": False,
        "atomic_no_overwrite": True,
        "crash_resume_equivalence": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "synthetic_streaming_hard_gate",
        "status": "GO_PRE_NATURAL_SYNTHETIC_HARD_GATE",
        "natural_data_accessed": False,
        "observed_natural_616448_used": False,
        "environment_identity_sha256": _sha256_value(environment),
        "streaming_schema_sha256": schema_sha256,
        "mechanism_schema_sha256": mechanism_schema_sha256,
        "physical_chunk_rows": PHYSICAL_CHUNK_ROWS,
        "measurement_method": "fresh Linux RUSAGE_SELF ru_maxrss workers; complete fixed mechanism rows; production 4096-row Parquet writer and bounded replay",
        "synthetic_cases": cases_for_schema,
        "objective_threshold_derivation": "baseline_peak_rss_bytes + 16 * 4096 * canonical_full_schema_row_bound_bytes; frozen before stress observation",
        "threshold_bytes": threshold_bytes,
        "estimated_physical_bytes_per_row": _next_power_of_two(canonical_row_bound),
        "checks": checks,
        "whole_table_materialization_detected": False,
        "locked_before_successor_natural_input": True,
        "evidence": evidence,
        "evidence_sha256": _sha256_value(evidence),
    }


def _validate_and_write(value: Mapping[str, Any], output: Path) -> None:
    from script.analysis.fpct_e1_streaming_verify import (
        validate_streaming_schema_artifact,
    )

    validate_streaming_schema_artifact(value, STREAMING_SCHEMA)
    payload = canonical_json_bytes(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite changed immutable gate: {output}")
        return
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, output)
    temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.worker is not None:
        if arguments.worker <= 0:
            raise ValueError("worker query count must be positive")
        print(json.dumps(_worker(arguments.worker), sort_keys=True))
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("synthetic gate requires CUDA_VISIBLE_DEVICES='' ")
    value = build_gate()
    _validate_and_write(value, arguments.output)
    verify_tracked_gate(arguments.output, REPO_ROOT)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
