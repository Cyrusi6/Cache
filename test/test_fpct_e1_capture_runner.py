from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import script.experiment.fpct_e1_capture_runner as runner


REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_IMAGE = "registry.example/fpct@sha256:" + "b" * 64


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(runner.canonical_json_bytes(value)).hexdigest()


def _synthetic_streaming_gate_fixture(repo_snapshot: Path) -> dict:
    """Build a cheap, fully bound gate receipt for capture-runner tests.

    The production million-row/RSS gate is exercised by its own test module.
    Capture-runner tests only need a schema-complete immutable receipt whose
    evidence, source, and test hashes can be independently recomputed.
    """

    from script.analysis import fpct_e1_streaming_synthetic_gate as gate

    environment = {
        "python": "fixture",
        "python_executable": "/fixture/python",
        "platform": "fixture-linux",
        "machine": "x86_64",
        "pyarrow": "fixture",
        "cuda_visible_devices": "",
    }
    baseline = {
        "case_id": "fixture_full_schema_two_chunk_baseline",
        "logical_rows": 4480,
        "emitted_rows": 4480,
        "semantic_stream_sha256": _sha("fixture-baseline-semantic-stream"),
        "replay_semantic_stream_sha256": _sha(
            "fixture-baseline-semantic-stream"
        ),
        "chunk_count": 2,
        "physical_file_bytes": 8192,
        "max_physical_chunk_bytes": 4096,
        "max_observed_canonical_row_bytes": 2048,
        "peak_rss_bytes": 64 * 1024 * 1024,
    }
    stress = {
        "case_id": "fixture_full_schema_million_row_stress",
        "logical_rows": 1_000_384,
        "emitted_rows": 1_000_384,
        "semantic_stream_sha256": _sha("fixture-stress-semantic-stream"),
        "replay_semantic_stream_sha256": _sha(
            "fixture-stress-semantic-stream"
        ),
        "chunk_count": 245,
        "physical_file_bytes": 2_048_000,
        "max_physical_chunk_bytes": 8192,
        "max_observed_canonical_row_bytes": 2048,
        "peak_rss_bytes": 96 * 1024 * 1024,
    }
    targeted_tests = {
        "command": ["fixture-pytest", *gate.TEST_FILES],
        "exit_code": 0,
        "output_sha256": _sha("fixture-targeted-tests-passed"),
        "last_output_line": "fixture targeted tests passed",
        "test_files": list(gate.TEST_FILES),
    }
    evidence = {
        "environment": environment,
        "targeted_tests": targeted_tests,
        "tracked_source_sha256": {
            relative: runner.sha256_file(repo_snapshot / relative)
            for relative in gate.TRACKED_SOURCE_FILES
        },
        "baseline_full_record": baseline,
        "stress_full_record": stress,
        "canonical_full_schema_row_bound_bytes": 2048,
        "buffer_copy_multiplier": gate.BUFFER_COPY_MULTIPLIER,
        "allocator_and_buffer_allowance_bytes": (
            gate.BUFFER_COPY_MULTIPLIER
            * gate.PHYSICAL_CHUNK_ROWS
            * 2048
        ),
    }
    checks = {
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
    }
    return {
        "schema_version": gate.SCHEMA_VERSION,
        "protocol_id": gate.PROTOCOL_ID,
        "artifact_type": "synthetic_streaming_hard_gate",
        "status": "GO_PRE_NATURAL_SYNTHETIC_HARD_GATE",
        "natural_data_accessed": False,
        "observed_natural_616448_used": False,
        "environment_identity_sha256": _canonical_sha(environment),
        "streaming_schema_sha256": runner.sha256_file(
            repo_snapshot / runner.STREAMING_SCHEMA_RELATIVE
        ),
        "mechanism_schema_sha256": runner.sha256_file(
            repo_snapshot / runner.MECHANISM_SCHEMA_RELATIVE
        ),
        "physical_chunk_rows": gate.PHYSICAL_CHUNK_ROWS,
        "measurement_method": (
            "capture-runner fixture with schema-complete synthetic records; "
            "the production million-row/RSS gate is not run by this fixture"
        ),
        "synthetic_cases": [
            {
                key: record[key]
                for key in (
                    "case_id",
                    "logical_rows",
                    "peak_rss_bytes",
                    "emitted_rows",
                    "semantic_stream_sha256",
                )
            }
            for record in (baseline, stress)
        ],
        "objective_threshold_derivation": (
            "fixture baseline plus the frozen 16 * 4096 * row-bound allowance"
        ),
        "threshold_bytes": (
            baseline["peak_rss_bytes"]
            + gate.BUFFER_COPY_MULTIPLIER
            * gate.PHYSICAL_CHUNK_ROWS
            * 2048
        ),
        "estimated_physical_bytes_per_row": 2048,
        "checks": checks,
        "whole_table_materialization_detected": False,
        "locked_before_successor_natural_input": True,
        "evidence": evidence,
        "evidence_sha256": _canonical_sha(evidence),
    }


def _gate(tmp_path: Path, repo_snapshot: Path, *, status: str = "GO") -> Path:
    evidence = repo_snapshot / "test/test_fpct_instrumentation.py"
    payload = {
        "schema_version": 1,
        "gate_id": runner.CONSOLIDATED_GATE_ID,
        "status": status,
        "checks": {name: True for name in runner.REQUIRED_GATE_CHECKS},
        "evidence": [{"logical_path": "repo://test/test_fpct_instrumentation.py", "sha256": runner.sha256_file(evidence)}],
    }
    path = tmp_path / "consolidated_gate.json"
    path.write_bytes(runner.canonical_json_bytes(payload))
    return path


def _checkpoint_root(tmp_path: Path) -> Path:
    root = tmp_path / "e0"
    data = root / "dev_data"
    data.mkdir(parents=True)
    (data / "frozen-input.parquet").write_bytes(b"synthetic-data")
    (root / "dev_data_manifest.json").write_bytes(runner.canonical_json_bytes({
        "schema_version": 1,
        "row_counts": runner.TASK_GROUP_COUNTS,
        "tree_sha256": runner._e0_declared_tree_sha256(data),
    }))
    for seed in runner.SEEDS:
        seed_root = root / "seeds" / str(seed)
        attempt = seed_root / "attempt_1"
        attempt.mkdir(parents=True)
        (seed_root / "active").symlink_to("attempt_1")
        for trained in runner.TRAINED_DIRECTORY.values():
            for leaf in ("checkpoint-64", "final"):
                directory = attempt / trained / leaf
                directory.mkdir(parents=True)
                (directory / "projector_0.json").write_text('{"version":1}\n')
                (directory / "projector_0.pt").write_bytes(b"synthetic-projector")
            (attempt / trained / "checkpoint-64" / "training_state.pt").write_bytes(b"state")
    return root


def _source_snapshot(tmp_path: Path) -> Path:
    from script.analysis import fpct_e1_streaming_synthetic_gate as gate

    repository = tmp_path / "source-repository"
    snapshot = tmp_path / "source-snapshot"
    relative_files = {
        runner.SPLIT_MANIFEST_RELATIVE,
        runner.E1_MANIFEST_RELATIVE,
        runner.STREAMING_CONTRACT_RELATIVE,
        runner.STREAMING_SCHEMA_RELATIVE,
        runner.STREAMING_SYNTHETIC_GATE_RELATIVE,
        runner.MECHANISM_SCHEMA_RELATIVE,
        runner.E0_DEV_MANIFEST_RELATIVE,
        runner.E0_CONFIG_INDEX_RELATIVE,
        runner.EXECUTOR_RELATIVE,
        runner.ANALYZER_RELATIVE,
        Path("script/experiment/fpct_e1_prepare_input_lock.py"),
        Path("script/experiment/fpct_e1_runtime_backend.py"),
        Path("script/experiment/fpct_e1_runtime_probe.py"),
        runner.RUNTIME_PROBE_RENDERER_RELATIVE,
        Path("script/experiment/fpct_e1_source_snapshot_lock.py"),
        Path("script/experiment/fpct_e1_k8s_lock_bundle.py"),
        Path("script/analysis/fpct_e1_streaming_verify.py"),
        Path("rosetta/model/wrapper.py"),
        Path("rosetta/model/fpct_attention.py"),
        Path("rosetta/model/fpct_instrumentation.py"),
        Path("rosetta/model/aligner.py"),
        Path("rosetta/train/dataset_adapters.py"),
        Path("rosetta/utils/evaluate.py"),
        Path("rosetta/utils/model_loading.py"),
        Path("script/evaluation/unified_evaluator.py"),
        Path("test/test_fpct_e1_capture_runner.py"),
        Path("test/test_fpct_e1_prepare_input_lock.py"),
        Path("test/test_fpct_e1_runtime_backend.py"),
        Path("test/test_fpct_e1_runtime_probe.py"),
        Path("test/test_fpct_e1_runtime_probe_k8s.py"),
        Path("test/test_fpct_e1_source_snapshot_lock.py"),
        Path("test/test_fpct_e1_k8s_lock_bundle.py"),
        Path("recipe/k8s/fpct_e1/runtime_probe_job.yaml"),
        runner.K8S_CAPTURE_TEMPLATE_RELATIVE,
        Path("test/test_fpct_e1_runtime_capture_primitives.py"),
        Path("test/test_fpct_instrumentation.py"),
        Path("test/test_fpct_e1_mechanism_audit.py"),
        Path("test/test_fpct_e1_streaming.py"),
    }
    relative_files.update(Path(path) for path in gate.TRACKED_SOURCE_FILES)
    relative_files.update(path.relative_to(REPO_ROOT) for path in (REPO_ROOT / runner.E0_CONFIG_ROOT_RELATIVE).glob("*"))
    for relative in sorted(relative_files):
        source, destination = REPO_ROOT / relative, repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative == runner.STREAMING_SYNTHETIC_GATE_RELATIVE and not source.exists():
            continue
        shutil.copy2(source, destination)
    manifest_path = repository / runner.E1_MANIFEST_RELATIVE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_authorization"] = {
        "a4_successor_input_lock_operative": True,
        "a4_successor_dag_operative": True,
        "requires_streaming_input_lock_go": True,
        "historical_attempt_resume_allowed": False,
        "e1_pilot_forward_or_outcome": False,
        "training": False,
    }
    manifest_path.write_bytes(runner.canonical_json_bytes(manifest))
    synthetic_path = repository / runner.STREAMING_SYNTHETIC_GATE_RELATIVE
    if not synthetic_path.exists():
        synthetic_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_path.write_bytes(
            runner.canonical_json_bytes(
                _synthetic_streaming_gate_fixture(repository)
            )
        )
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), *args], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        return result.stdout.strip()
    git("init", "-q")
    git("config", "user.email", "fpct-runner-test@example.invalid")
    git("config", "user.name", "FPCT Runner Test")
    git("add", ".")
    environment = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2026-07-26T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2026-07-26T00:00:00+00:00",
    }
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "snapshot"],
        check=True, env=environment,
    )
    execution_sha = git("rev-parse", "HEAD")
    archive_path = tmp_path / "source-snapshot.tar"
    subprocess.run(
        ["git", "-C", str(repository), "archive", "--format=tar", f"--output={archive_path}", execution_sha],
        check=True,
    )
    snapshot.mkdir()
    with tarfile.open(archive_path, "r:") as archive:
        archive.extractall(snapshot)
    from script.experiment.fpct_e1_source_snapshot_lock import (
        DEFAULT_RECEIPT_NAME, create_source_snapshot_lock,
    )
    create_source_snapshot_lock(
        repo=repository, execution_sha=execution_sha, snapshot_root=snapshot,
        output_path=snapshot / DEFAULT_RECEIPT_NAME,
    )
    return snapshot


def _snapshot_receipt(snapshot: Path) -> tuple[Path, dict]:
    from script.experiment.fpct_e1_source_snapshot_lock import DEFAULT_RECEIPT_NAME

    path = snapshot / DEFAULT_RECEIPT_NAME
    return path, json.loads(path.read_text(encoding="utf-8"))


def _resolved_plan(tmp_path: Path) -> tuple[dict, Path, Path]:
    repo_snapshot = _source_snapshot(tmp_path)
    source_receipt, receipt_payload = _snapshot_receipt(repo_snapshot)
    execution_sha = receipt_payload["execution_sha"]
    e0_root = _checkpoint_root(tmp_path)
    gate = _gate(tmp_path, repo_snapshot)
    from script.experiment.fpct_e1_source_snapshot_lock import (
        verify_source_snapshot_receipt,
    )
    source_receipt_raw = source_receipt.read_bytes()
    source_verification = {
        **verify_source_snapshot_receipt(
            source_receipt, repo_snapshot, execution_sha,
        ),
        "receipt_file_sha256": runner.sha256_bytes(source_receipt_raw),
        "receipt_bytes": len(source_receipt_raw),
    }
    runtime = tmp_path / "runtime.json"
    runtime.write_bytes(runner.canonical_json_bytes({
        "schema_version": 1,
        "protocol_id": "fpct_e1_runtime_probe_v1",
        "status": "FROZEN_MODEL_OUTPUT_FREE_RUNTIME_PROBE",
        "execution_sha": execution_sha,
        "image_digest": FAKE_IMAGE,
        "source_snapshot_tree_sha": receipt_payload["snapshot"]["mounted_tree_canonical_sha256"],
        "source_snapshot_verification": source_verification,
        "runtime": {
            "python": {"version": "3.10", "implementation": "CPython", "executable": "/python"},
            "platform": {"system": "Linux", "release": "test", "version": "test", "machine": "x86_64", "platform": "test"},
            "packages": {"torch": "test", "transformers": "test", "tokenizers": "test", "pyarrow": "test"},
            "torch_cuda_build": "12.4",
            "cuda": {"available": False, "initialized_before_probe": False, "initialized_after_probe": False, "cuda_visible_devices": "", "device_count": 0, "devices": []},
        },
        "offline": {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "enforced_before_torch_import": True, "network_download_attempted": False},
        "firewall": {"model_loaded": False, "tokenizer_loaded": False, "checkpoint_loaded": False, "model_output_accessed": False},
    }))
    common = dict(
        instrumentation_gate_path=gate,
        execution_sha=execution_sha,
        image_digest=FAKE_IMAGE,
        backend_spec="fixture:backend",
        backend_source=repo_snapshot / "test/test_fpct_e1_capture_runner.py",
        runtime_provenance=runtime,
        repo_container="/container/repo",
        e0_container="/container/e0",
        output_container="/container/output",
        input_lock_container="/container/input-lock",
        raw_topology_container="/container/raw-topology",
        source_snapshot_receipt=source_receipt,
    )
    draft = runner.build_execution_plan(
        repo_snapshot,
        e0_root,
        tmp_path / "outputs",
        **common,
    )
    input_lock_root = tmp_path / "input-lock"
    input_lock_root.mkdir()
    sidecar = input_lock_root / "e1_input_lock.pt"
    execution_identity = {
        "schema_version": 1,
        "protocol_id": "fpct_e1_a4_input_lock_execution_identity_v1",
        "execution_sha": execution_sha,
        "execution_prefix": execution_sha[:8],
        "run_uid": f"fpct-e1-a4-streaming-{execution_sha[:8]}-v1",
        "run_root": str(tmp_path.absolute()),
        "source_snapshot_root": str(repo_snapshot.absolute()),
        "source_snapshot_receipt": {
            "path": str(source_receipt.absolute()),
            "bytes": source_receipt.stat().st_size,
            "file_sha256": runner.sha256_file(source_receipt),
        },
        "input_lock_root": str(input_lock_root.absolute()),
        "historical_execution_resume_allowed": False,
        "historical_artifact_reuse_allowed": False,
        "identity_sha256": _sha("a4-identity:" + execution_sha),
    }
    dimensions = {"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8}
    schema_sha = runner.sha256_file(repo_snapshot / runner.STREAMING_SCHEMA_RELATIVE)
    items = []
    for task in runner.TASKS:
        for group in draft["group_contract"][task]:
            answer_queries = [
                {
                    "query_position": 4,
                    "target_position": 5,
                    "target_token_id": 7,
                }
            ]
            certified_parents = [{
                "parent_position": 2,
                "candidate_count": 2,
                "prior": [0.6, 0.4],
                "topology": "partition_compositional",
                "candidate_indices": [3, 4, -1, -1],
                "candidate_valid_mask": [True, True, False, False],
                "candidate_slot_weights": [0.6, 0.4, 0.0, 0.0],
                "statistical_weight": 1.0,
            }]
            raw_topology_compact = {
                "raw_m_ge_2_parent_count": 0,
                "raw_parent_sequence_sha256": _sha(
                    "raw-topology-empty:" + group["content_group_sha256"]
                ),
            }
            item = {
                "task": task,
                "sample_sha256": group["sample_sha256"][0],
                "content_group_sha256": group["content_group_sha256"],
                "descriptor": {
                    "rendered_prompt_sha256": group["rendered_prompt_sha256"],
                    "prompt_alignment_sha256": group["prompt_alignment_sha256"],
                },
                "rendered_prompt_sha256": group["rendered_prompt_sha256"],
                "provenance": {
                    "input_sha256": _sha("input:" + group["content_group_sha256"]),
                    "alignment_sha256": _sha("full-response-alignment:" + group["content_group_sha256"]),
                    "labels_sha256": _sha("labels:" + group["content_group_sha256"]),
                    "gold_response_sha256": _sha("gold:" + group["content_group_sha256"]),
                },
                "answer_queries": answer_queries,
                "certified_parents": certified_parents,
                "Q_s": 1,
                "P_s": 1,
                "N_s": 28 * 16,
                "answer_query_sequence_sha256": runner._nested_semantic_sha256(
                    answer_queries
                ),
                "parent_sequence_sha256": runner._nested_semantic_sha256(
                    certified_parents
                ),
                "raw_topology_compact": raw_topology_compact,
                "raw_topology_compact_sha256": runner._nested_semantic_sha256(
                    raw_topology_compact
                ),
                "expected_chunk_count": 1,
                "expected_long_form_rows": 28 * 16,
            }
            item["item_semantic_sha256"] = runner._nested_semantic_sha256(item)
            items.append(item)
    from script.experiment.fpct_e1_prepare_input_lock import raw_topology_ledger
    topology_item = items[0]
    topology_item["raw_topology_ledger"] = raw_topology_ledger(
        raw_details={
            "message_mask": [True],
            "slm_ids": [101],
            "llm_ids": [201, 202],
            "slm_offsets": [(0, 2)],
            "llm_offsets": [(0, 1), (1, 2)],
            "content_spans_slm": [(0, 2)],
            "content_spans_llm": [(0, 2)],
            "sections": [{"type": "message", "slm_range": (0, 1), "llm_range": (0, 2)}],
            "soft_alignment": {"source_indices": [[0, 1]], "source_weights": [[0.5, 0.5]]},
        },
        sanitized_details={
            "soft_alignment": {
                "source_indices": [[0, 1]],
                "source_weights": [[0.5, 0.5]],
                "fpct_certified_mask": [True],
                "fpct_certification_reason": ["certified_disjoint_partition"],
            }
        },
        instruction_end=1,
        task=topology_item["task"],
        sample_sha256=topology_item["sample_sha256"],
        content_group_sha256=topology_item["content_group_sha256"],
        candidate_window=0,
    )
    topology_item["raw_topology_compact"] = {
        "raw_m_ge_2_parent_count": len(topology_item["raw_topology_ledger"]),
        "raw_parent_sequence_sha256": runner._nested_semantic_sha256(
            topology_item["raw_topology_ledger"]
        ),
    }
    topology_item["raw_topology_compact_sha256"] = runner._nested_semantic_sha256(
        topology_item["raw_topology_compact"]
    )
    topology_item["item_semantic_sha256"] = runner._nested_semantic_sha256(
        {key: value for key, value in topology_item.items() if key != "item_semantic_sha256"}
    )
    sidecar_payload = {
        "schema_version": runner.INPUT_LOCK_SCHEMA_VERSION,
        "protocol_id": runner.INPUT_LOCK_PROTOCOL_ID,
        "status": runner.INPUT_LOCK_STATUS,
        "split_role": runner.ALLOWED_SPLIT_ROLE,
        "items": items,
        "dimensions": dimensions,
        "execution_identity": execution_identity,
        "firewall": {
            "e1_pilot_consumed": False,
            "model_selection_consumed": False,
            "test_consumed": False,
            "model_or_checkpoint_loaded": False,
            "gpu_or_cuda_used": False,
        },
        "e1_pilot_consumed": False,
        "model_or_checkpoint_loaded": False,
        "cuda_initialized": False,
    }
    torch.save(sidecar_payload, sidecar)
    task_contract = {}
    from script.analysis.fpct_e1_streaming_verify import (
        SampleRowStream,
        PARQUET_MANIFEST_NAME,
        attest_ordered_sample_streams,
        canonical_endpoint_id,
        write_parquet_stream_artifact,
    )
    for task in runner.TASKS:
        members = sorted((item for item in items if item["task"] == task), key=lambda item: item["sample_sha256"])
        streams = [
            SampleRowStream(
                item,
                num_layers=28,
                num_query_heads=16,
                num_kv_heads=8,
                endpoint_id=canonical_endpoint_id(0, "input_geometry", "input_geometry", "INPUT_LOCK", task, 0.0),
                schema_sha256=schema_sha,
            )
            for item in members
        ]
        attested = attest_ordered_sample_streams(streams)
        task_contract[task] = {
            "group_count": len(draft["group_contract"][task]),
            "members": [{"sample_sha256": item["sample_sha256"], "content_group_sha256": item["content_group_sha256"], "item_semantic_sha256": item["item_semantic_sha256"]} for item in members],
            "membership_sha256": runner._nested_semantic_sha256(sorted([item["sample_sha256"], item["content_group_sha256"], item["item_semantic_sha256"]] for item in members)),
            "row_template": {
                "count": attested["logical_row_count"],
                "semantic_stream_sha256": attested["semantic_stream_sha256"],
                "schema_sha256": schema_sha,
                "physical_chunk_rows": runner.PARQUET_BATCH_ROWS,
                "sample_count": attested["sample_count"],
                "first_sample_sha256": attested["first_sample_sha256"],
                "last_sample_sha256": attested["last_sample_sha256"],
                "ordinal_order": "sample_sha256,row_ordinal",
            },
            "expected_long_form_rows": runner._expected_long_form_rows_contract(members),
        }
    index_records = []
    global_template_digest = hashlib.sha256()
    for item in sorted(items, key=lambda value: (value["sample_sha256"], value["content_group_sha256"])):
        stream = SampleRowStream(
            item, num_layers=28, num_query_heads=16, num_kv_heads=8,
            endpoint_id=canonical_endpoint_id(0, "input_geometry", "input_geometry", "INPUT_LOCK", item["task"], 0.0),
            schema_sha256=schema_sha,
        )
        sample_root = input_lock_root / "row_templates" / item["sample_sha256"]
        sample_root.mkdir(parents=True, exist_ok=True)
        execution_binding = sample_root / "execution_binding.json"
        execution_binding.write_bytes(runner.canonical_json_bytes({
            "schema_version": 1,
            "protocol_id": "fpct_e1_a4_sample_stream_execution_binding_v1",
            "execution_identity_sha256": execution_identity["identity_sha256"],
            "sample_sha256": item["sample_sha256"],
            "item_semantic_sha256": item["item_semantic_sha256"],
            "historical_artifact_reuse_allowed": False,
        }))
        verified = write_parquet_stream_artifact(sample_root, stream)
        sample_manifest = sample_root / PARQUET_MANIFEST_NAME
        for row in stream.iter_rows():
            global_template_digest.update(runner.canonical_json_bytes(row))
        index_records.append({
            "task": item["task"], "sample_sha256": item["sample_sha256"],
            "content_group_sha256": item["content_group_sha256"],
            "manifest_relative_path": str(sample_manifest.relative_to(input_lock_root)),
            "manifest_sha256": runner.sha256_file(sample_manifest),
            "manifest_bytes": sample_manifest.stat().st_size,
            "expected_logical_rows": stream.row_count,
            "emitted_logical_rows": verified["emitted_logical_rows"],
            "chunk_count": verified["chunk_count"],
            "semantic_stream_sha256": verified["semantic_stream_sha256"],
            "item_semantic_sha256": item["item_semantic_sha256"],
            "raw_topology_compact_sha256": item[
                "raw_topology_compact_sha256"
            ],
            "execution_binding_relative_path": str(
                execution_binding.relative_to(input_lock_root)
            ),
            "execution_binding_sha256": runner.sha256_file(execution_binding),
        })
    geometry_samples = input_lock_root / "input_geometry_samples.parquet"
    geometry_samples.write_bytes(b"synthetic-compact-geometry")
    geometry_manifest = input_lock_root / "input_geometry_manifest.json"
    geometry_manifest.write_bytes(runner.canonical_json_bytes({"schema_version": 6, "protocol_id": runner.A4_PROTOCOL_ID, "artifact_type": "input_geometry_manifest"}))
    geometry_receipt = input_lock_root / "input_geometry_receipt.json"
    geometry_receipt.write_bytes(runner.canonical_json_bytes({
        "schema_version": 6, "protocol_id": runner.A4_PROTOCOL_ID,
        "status": "GO", "expanded_logical_rows_materialized": False,
        "e1_pilot_consumed": False, "model_or_checkpoint_loaded": False,
        "gpu_or_kubernetes_used": False,
    }))
    geometry_lock = {
        name: {"path": str(path), "sha256": runner.sha256_file(path), "bytes": path.stat().st_size}
        for name, path in {"manifest": geometry_manifest, "samples": geometry_samples, "receipt": geometry_receipt}.items()
    }
    total_rows = sum(item["N_s"] for item in items)
    template_index = input_lock_root / "input_row_template_chunk_index.json"
    template_index.write_bytes(runner.canonical_json_bytes({
        "schema_version": 6, "protocol_id": runner.A4_PROTOCOL_ID,
        "artifact_type": "chunk_manifest_index", "population": "e0_design",
        "physical_chunk_rows": runner.PARQUET_BATCH_ROWS,
        "sample_count": len(items), "expected_logical_rows": total_rows,
        "emitted_logical_rows": total_rows,
        "semantic_stream_sha256": global_template_digest.hexdigest(),
        "schema_sha256": schema_sha, "records": index_records,
    }))
    template_receipt = input_lock_root / "streaming_input_lock_receipt.json"
    receipt_true = {
        name: True for name in (
            "protocol_and_schema_versioned", "expected_logical_rows_eq_emitted",
            "ordinal_first_eq_zero", "ordinal_last_eq_expected_minus_one",
            "ordinal_ranges_contiguous", "row_key_reference_equivalence",
            "weights_reference_equivalence", "topology_reference_equivalence",
            "semantic_stream_replay_equal", "chunk_partition_semantic_equivalence",
            "aggregate_partition_equivalence", "bounded_peak_rss",
            "atomic_no_overwrite", "crash_resume_equivalence",
            "raw_topology_complete", "task_counts_exact_128_70_128",
            "input_assets_unchanged_during_lock",
        )
    }
    template_receipt.write_bytes(runner.canonical_json_bytes({
        "schema_version": 6, "protocol_id": runner.A4_PROTOCOL_ID,
        "status": "GO", **receipt_true,
        "old_attempt_artifact_reused": False, "e1_pilot_consumed": False,
        "model_or_checkpoint_loaded": False, "gpu_or_kubernetes_used": False,
        "whole_table_materialization_detected": False,
        "missing_rows": 0, "duplicate_rows": 0, "overlapping_chunks": 0,
        "chunk_manifest_index_sha256": runner.sha256_file(template_index),
        "geometry_manifest_sha256": runner.sha256_file(geometry_manifest),
        "semantic_stream_sha256": global_template_digest.hexdigest(),
        "schema_sha256": schema_sha,
        "rss_lock_sha256": runner.sha256_file(repo_snapshot / runner.STREAMING_SYNTHETIC_GATE_RELATIVE),
    }))
    template_lock = {
        "index": {"path": str(template_index), "sha256": runner.sha256_file(template_index), "bytes": template_index.stat().st_size},
        "receipt": {"path": str(template_receipt), "sha256": runner.sha256_file(template_receipt), "bytes": template_receipt.stat().st_size},
        "expected_logical_rows": total_rows, "emitted_logical_rows": total_rows,
        "semantic_stream_sha256": global_template_digest.hexdigest(),
    }
    sidecar_payload["streaming_contract"] = {
        "protocol_id": runner.A4_PROTOCOL_ID,
        "schema_sha256": schema_sha,
        "physical_chunk_rows": runner.PARQUET_BATCH_ROWS,
        "expanded_logical_rows_present": False,
        "compact_geometry_only": True,
        "geometry_lock": geometry_lock,
        "streaming_template_lock": template_lock,
    }
    torch.save(sidecar_payload, sidecar)
    input_manifest = input_lock_root / "e1_input_lock_manifest.json"
    input_manifest.write_bytes(runner.canonical_json_bytes({
        "schema_version": runner.INPUT_LOCK_SCHEMA_VERSION,
        "protocol_id": runner.INPUT_LOCK_PROTOCOL_ID,
        "status": runner.INPUT_LOCK_STATUS,
        "split_role": runner.ALLOWED_SPLIT_ROLE,
        "execution_identity": execution_identity,
        "dimensions": dimensions,
        "task_contract": task_contract,
        "expected_long_form_rows_by_task": {
            task: {
                "count": task_contract[task]["expected_long_form_rows"]["count"],
                "sum": task_contract[task]["expected_long_form_rows"]["sum"],
            }
            for task in runner.TASKS
        },
        "runtime_assets": {
            role: {
                "model_id": model_id,
                "requested_path": f"/models/{role}",
                "resolved_path": f"/models/{role}",
                "root_kind": "directory",
                "root_symlink_target": None,
                "files": [{"relative_path": "config.json", "kind": "file", "symlink_target": None, "bytes": 2, "sha256": _sha(role + ":config")}],
                "file_count": 1,
                "bytes": 2,
                "tree_sha256": _sha(role + ":tree"),
            }
            for role, model_id in {"receiver": "Qwen/Qwen3-0.6B", "sender": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}.items()
        },
        "streaming_contract": {
            "protocol_id": runner.A4_PROTOCOL_ID,
            "schema_sha256": schema_sha,
            "physical_chunk_rows": runner.PARQUET_BATCH_ROWS,
            "historical_cumulative_ceiling_operative": False,
            "geometry_lock": geometry_lock,
            "streaming_template_lock": template_lock,
            "synthetic_gate": {
                "path": str(repo_snapshot / runner.STREAMING_SYNTHETIC_GATE_RELATIVE),
                "sha256": runner.sha256_file(repo_snapshot / runner.STREAMING_SYNTHETIC_GATE_RELATIVE),
            },
        },
        "sidecar": {"path": str(sidecar), "bytes": sidecar.stat().st_size, "sha256": runner.sha256_file(sidecar)},
        "firewall": {"e1_pilot_consumed": False, "model_selection_consumed": False, "test_consumed": False, "model_or_checkpoint_loaded": False, "gpu_or_cuda_used": False},
    }))
    from script.analysis.fpct_e1_mechanism_audit import write_raw_topology_artifacts
    raw_topology_root = tmp_path / "raw-topology"
    raw_topology_manifest = write_raw_topology_artifacts(sidecar, raw_topology_root)
    plan = runner.build_execution_plan(
        repo_snapshot,
        e0_root,
        tmp_path / "outputs",
        input_lock_manifest=input_manifest,
        input_lock_sidecar=sidecar,
        raw_topology_manifest=raw_topology_root / "e1_raw_topology_manifest.json",
        raw_topology_artifact_root=raw_topology_root,
        **common,
    )
    path = tmp_path / "capture_plan.json"
    path.write_bytes(runner.canonical_json_bytes(plan))
    return plan, path, gate


def _find_shard(plan: dict, *, seed: int, arm: str, task: str, value: float, stage: str) -> dict:
    matches = [
        row for row in plan["shards"]
        if row["seed"] == seed and row["checkpoint_arm"] == arm and row["task"] == task
        and row["lambda_value"] == value and row["stage"] == stage
    ]
    assert len(matches) == 1
    return matches[0]


def _execute_shard(
    plan_path: Path, shard_id: str, gate_path: Path | None,
    output_root: Path, backend_spec: str | None, **kwargs,
):
    """Invoke a shard with the same mechanical identities as rendered K8s."""

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    kwargs.pop("claim_id", None)
    kwargs.setdefault("expected_plan_sha256", plan["plan_sha256"])
    kwargs["claim_id"] = runner._expected_claim_id(plan["plan_sha256"], shard_id)
    return runner.execute_shard(
        plan_path, shard_id, gate_path, output_root, backend_spec, **kwargs,
    )


def _write_phase_marker(plan: dict, output: Path, stage: str, closure: str = "4" * 64) -> Path:
    count = {runner.PHASE_E1_2_BASELINES: 18, runner.PHASE_E1_2_FACTORIZED: 18, runner.STAGE_E1_2: 36, runner.STAGE_E1_3: 108}[stage]
    marker = {"schema_version": 1, "status": "GO", "stage": stage, "plan_sha256": plan["plan_sha256"], "shard_count": count, "closure_sha256": closure}
    path = runner._phase_marker(output, stage)
    runner.atomic_write(path, runner.canonical_json_bytes(marker))
    return path


def _row(shard: dict, identity: dict) -> dict:
    value = shard["lambda_value"]
    prior = [0.6, 0.4]
    gamma = prior if value == 0 else [0.5 + 0.1 * min(value, 2), 0.5 - 0.1 * min(value, 2)]
    return {
        "schema_version": runner.SCHEMA_VERSION,
        "split_role": runner.ALLOWED_SPLIT_ROLE,
        "seed": shard["seed"],
        "checkpoint_arm": shard["checkpoint_arm"],
        "inference_operator": shard["inference_operator"],
        "cell": shard["cell"],
        "task": shard["task"],
        **{name: identity[name] for name in (
            *runner.ROW_TEMPLATE_COLUMNS, "row_ordinal", "logical_row_id",
            "endpoint_id", "endpoint_row_id", "candidate_indices",
            "candidate_valid_mask", "candidate_slot_weights",
            "statistical_weight",
        )},
        "lambda_value": value,
        "prior": identity["prior"],
        "gamma": gamma,
        "source_d_k": 0.4,
        "source_d_v": 0.3,
        "source_energy_k": 2.0,
        "source_energy_v": 1.5,
        "fused_d_k": 0.1,
        "fused_d_v": 0.15,
        "fused_energy_k": 1.0,
        "fused_energy_v": 1.0,
        "candidate_logit_range": 0.0 if value == 0 else 0.2 * value,
        "candidate_logit_variance": 0.0 if value == 0 else 0.01 * value,
        "jensen_gap": 0.0 if value == 0 else 0.005 * value,
        "parent_attention_mass": 0.25,
        "output_delta_l2": 0.0 if value == 0 else 0.01 * value,
        "gold_logp": -1.25 if value == 0 else -1.25 + 0.01 * value,
        "end_task_correct": value != 2.0,
    }


def _sorted_rows(plan: dict, shard: dict) -> list[dict]:
    streams, _ = runner._capture_sample_streams(plan, shard, "host")
    return [
        _row(shard, stream.row_at(ordinal))
        for sample in sorted(streams)
        for stream in (streams[sample],)
        for ordinal in range(stream.row_count)
    ]


def _first_parquet_row(path: Path) -> dict:
    import pyarrow.parquet as pq

    batch = next(pq.ParquetFile(path).iter_batches(batch_size=1))
    return batch.to_pylist()[0]


def _first_capture_row(output_dir: Path) -> dict:
    return next(runner._iter_capture_parquet_rows(output_dir))


def _fixture_stable_attestation(plan: dict, shard: dict) -> dict:
    checkpoint = next(
        record for record in plan["checkpoints"]
        if record["checkpoint_id"] == shard["checkpoint_id"]
    )
    files = checkpoint["final"]["files"]
    json_file = next(row for row in files if row["relative"].endswith("projector_0.json"))
    state_file = next(row for row in files if row["relative"].endswith("projector_0.pt"))
    runtime_assets = {
        role: {
            "model_id": frozen["model_id"],
            "runtime_path": frozen["resolved_path"],
            "tree_sha256": frozen["tree_sha256"],
            "file_count": frozen["file_count"],
            "bytes": frozen["bytes"],
            "verified_before_model_or_tokenizer_load": True,
        }
        for role, frozen in plan["input_lock"]["prepared"]["runtime_assets"].items()
    }
    expected_rows = {
        **plan["input_lock"]["prepared"]["task_contract"][shard["task"]]
        ["expected_long_form_rows"],
        "source": "frozen_cpu_input_lock",
        "passed_explicitly_per_item": True,
        "exact_row_count_required": True,
        "verified_before_model_load": True,
    }
    return {
        "split_role": runner.ALLOWED_SPLIT_ROLE,
        "capture_mode": "teacher_forced_response",
        "causal_shift_verified": True,
        "stores_raw_kv": False,
        "long_form_contract_version": 2,
        "plan_sha256": plan["plan_sha256"],
        "shard_id": shard["shard_id"],
        "execution_sha": plan["runtime_lock"]["execution_sha"],
        "image_digest": plan["runtime_lock"]["image_digest"],
        "checkpoint_tree_sha256": shard["checkpoint_tree_sha256"],
        "membership_sha256": shard["membership_sha256"],
        "gold_response_template": runner.GOLD_RESPONSE_TEMPLATE,
        "gold_response_template_sha256": runner.sha256_bytes(
            runner.GOLD_RESPONSE_TEMPLATE.encode("utf-8")
        ),
        "correctness_semantics": runner.CORRECTNESS_SEMANTICS,
        "backend_supplied_cpost_fields": False,
        "runtime_assets": runtime_assets,
        "projector_load": {
            "mode": "strict_attested",
            "record_count": 1,
            "records": [{
                "source_model_index": 1,
                "projector_index": 0,
                "json_path": f"/fixture/projector_0.json",
                "json_sha256": json_file["sha256"],
                "state_path": f"/fixture/projector_0.pt",
                "state_sha256": state_file["sha256"],
                "strict": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "state_key_count": 1,
            }],
        },
        "expected_long_form_rows": expected_rows,
        "e1_pilot_consumed": False,
        "model_selection_consumed": False,
        "test_consumed": False,
    }


def _backend_for(plan: dict, shard: dict, *, inject_cpost: bool = False, drop_last: bool = False):
    def backend(request):
        assert request["schema_boundary"]["forbidden_prefix"] == "cpost_"
        assert Path(request["input_lock_manifest_path"]).is_file()
        assert Path(request["input_lock_sidecar_path"]).is_file()
        assert request["expected_row_template"] == shard["expected_row_template"]
        assert request["input_lock_manifest_sha256"] == plan["input_lock"]["prepared"]["manifest"]["sha256"]
        assert request["input_lock_sidecar_sha256"] == plan["input_lock"]["prepared"]["sidecar"]["sha256"]
        assert request["runtime_assets"] == plan["input_lock"]["prepared"]["runtime_assets"]
        rows = _sorted_rows(plan, shard)
        cursor = request["resume_cursor"]
        cursor_payload = {
            key: value for key, value in cursor.items()
            if key != "cursor_sha256"
        }
        assert cursor["cursor_sha256"] == runner.sha256_bytes(
            runner.canonical_json_bytes(cursor_payload)
        )
        completed = cursor["completed_logical_rows"]
        assert 0 <= completed <= len(rows)
        if completed < len(rows):
            assert (
                rows[completed]["sample_sha256"],
                rows[completed]["row_ordinal"],
            ) == (
                cursor["next_sample_sha256"],
                cursor["next_row_ordinal"],
            )
        else:
            assert cursor["next_sample_sha256"] is None
        rows = rows[completed:]
        if inject_cpost:
            rows[0]["cpost_gold_logp"] = 999.0
        if drop_last:
            rows = rows[:-1]
        return {
            "contract_version": runner.CAPTURE_BACKEND_CONTRACT_VERSION,
            "rows": iter(rows),
            "resume_prefix_verification": {
                "schema_version": 1,
                "protocol_id": runner.CAPTURE_STAGING_PROTOCOL_ID,
                "cursor_sha256": cursor["cursor_sha256"],
                "expected_row_count": cursor[
                    "partial_sample_backend_prefix_row_count"
                ],
                "expected_backend_projection_sha256": cursor[
                    "partial_sample_backend_prefix_sha256"
                ],
                "status": "GO_EXACT_PREFIX_MATCH",
                "observed_row_count": cursor[
                    "partial_sample_backend_prefix_row_count"
                ],
                "observed_backend_projection_sha256": cursor[
                    "partial_sample_backend_prefix_sha256"
                ],
                "exact_match": True,
            },
            "attestation": {
                **_fixture_stable_attestation(plan, shard),
                "resume_cursor_sha256": cursor["cursor_sha256"],
                "resume_completed_logical_rows": completed,
                "resume_next_sample_sha256": cursor["next_sample_sha256"],
                "resume_next_row_ordinal": cursor["next_row_ordinal"],
            },
        }
    return backend


def test_plan_freezes_two_stage_36_plus_72_graph_and_lambda_ids(tmp_path: Path) -> None:
    plan, _, _ = _resolved_plan(tmp_path)
    runner.validate_execution_plan(plan, require_checkpoints=True, for_execution=True)
    assert plan["shard_count"] == runner.EXPECTED_SHARD_COUNT == 108
    assert plan["schema_version"] == 6
    assert plan["protocol_id"] == runner.PROTOCOL_ID
    assert plan["a4_operative_lock"]["status"] == "GO_NEW_A4_SUCCESSOR_DAG_ONLY"
    assert plan["a4_operative_lock"]["historical_execution_resume_allowed"] is False
    assert plan["input_lock"]["prepared"]["status"] == runner.INPUT_LOCK_STATUS
    assert sum(row["stage"] == runner.STAGE_E1_2 for row in plan["shards"]) == 36
    assert sum(row["stage"] == runner.STAGE_E1_3 for row in plan["shards"]) == 72
    assert all("lambda-" in row["shard_id"] for row in plan["shards"])
    assert all(row["requires_phase_completion"] == runner.FINALIZED_E1_2 for row in plan["shards"] if row["stage"] == runner.STAGE_E1_3)
    assert sum(row["phase"] == runner.PHASE_E1_2_BASELINES for row in plan["shards"]) == 18
    assert sum(row["phase"] == runner.PHASE_E1_2_FACTORIZED for row in plan["shards"]) == 18
    lambda_zero_controls = [row for row in plan["shards"] if row["stage"] == runner.STAGE_E1_3 and row["lambda_value"] == 0.0]
    assert len(lambda_zero_controls) == 18
    assert all(row["inference_operator"] == "f" and row["depends_on_shard"] for row in lambda_zero_controls)
    for seed in runner.SEEDS:
        for arm in runner.CHECKPOINT_ARMS:
            for task in runner.TASKS:
                values = {row["lambda_value"] for row in plan["shards"] if row["seed"] == seed and row["checkpoint_arm"] == arm and row["task"] == task}
                assert values == set(runner.FULL_LAMBDA_GRID)
    frozen_sources = {record["logical_path"] for record in plan["source_files"]}
    for relative in (
        "script/experiment/fpct_e1_prepare_input_lock.py",
        "script/experiment/fpct_e1_runtime_backend.py",
        "script/experiment/fpct_e1_runtime_probe_renderer.py",
        "rosetta/model/wrapper.py",
        "rosetta/model/fpct_attention.py",
        "rosetta/model/fpct_instrumentation.py",
        "rosetta/model/aligner.py",
        "rosetta/train/dataset_adapters.py",
        "rosetta/utils/evaluate.py",
        "rosetta/utils/model_loading.py",
        "script/evaluation/unified_evaluator.py",
        "test/test_fpct_e1_prepare_input_lock.py",
        "test/test_fpct_e1_runtime_backend.py",
        "test/test_fpct_e1_runtime_capture_primitives.py",
        "script/analysis/fpct_e1_streaming_verify.py",
        "test/test_fpct_e1_streaming.py",
        "recipe/eval_recipe/fpct_e1/e1_streaming_contract.json",
        "recipe/eval_recipe/fpct_e1/e1_streaming_schema.json",
        "recipe/eval_recipe/fpct_e1/e1_streaming_synthetic_gate.json",
    ):
        assert f"repo://{relative}" in frozen_sources
    assert set(runner._row_key_columns(runner._audit_columns()[1])).isdisjoint(
        {"row_ordinal", "logical_row_id", "endpoint_id", "endpoint_row_id"}
    )


def test_runtime_probe_renderer_is_frozen_and_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    plan, _, _ = _resolved_plan(tmp_path)
    logical = f"repo://{runner.RUNTIME_PROBE_RENDERER_RELATIVE.as_posix()}"
    records = {
        record["logical_path"]: record for record in plan["source_files"]
    }
    assert logical in records
    renderer_path = runner.resolve_logical_path(plan, logical)
    assert records[logical]["sha256"] == runner.sha256_file(renderer_path)

    original = renderer_path.read_bytes()
    renderer_path.write_bytes(original + b"\n# provenance tamper\n")
    try:
        with pytest.raises(ValueError, match="provenance changed"):
            runner.verify_plan_sources(plan)
    finally:
        renderer_path.write_bytes(original)


def test_active_symlink_resolves_to_immutable_attempt_and_binds_step64_final(tmp_path: Path) -> None:
    plan, _, _ = _resolved_plan(tmp_path)
    for record in plan["checkpoints"]:
        assert record["immutable_attempt"] == "attempt_1"
        assert "/active/" not in record["final"]["logical_path"]
        assert record["final"]["tree_sha256"] != record["step64"]["tree_sha256"]
        assert runner._is_sha256(record["projector_artifact_set_sha256"])
        assert record["final"]["files"]
        assert record["step64"]["files"]


def test_portable_logical_paths_map_to_host_and_container(tmp_path: Path) -> None:
    plan, _, _ = _resolved_plan(tmp_path)
    record = plan["checkpoints"][0]["final"]["logical_path"]
    assert str(runner.resolve_logical_path(plan, record, "host")).startswith(str(tmp_path / "e0"))
    assert str(runner.resolve_logical_path(plan, record, "container")).startswith("/container/e0")
    prepared_sidecar = plan["input_lock"]["prepared"]["sidecar"]["logical_path"]
    assert str(runner.resolve_logical_path(plan, prepared_sidecar, "container")).startswith("/container/input-lock/")
    assert str(runner.resolve_logical_path(plan, plan["raw_topology_lock"]["root"], "container")).startswith("/container/raw-topology")
    assert plan["raw_topology_lock"]["status"] == "GO_MODEL_OUTPUT_FREE"
    assert all("checkpoint_path" not in shard for shard in plan["shards"])
    assert all("prompt_alignment_sha256" in group and "alignment_sha256" not in group for groups in plan["group_contract"].values() for group in groups)
    declared = plan["input_lock"]["manifest"]
    assert declared["declared_tree_sha256"] == declared["recomputed_declared_tree_sha256"]
    assert declared["declared_tree_algorithm"] == "relative_path_nul_file_sha256_bytes_v1"
    assert declared["declared_tree_sha256"] != plan["input_lock"]["tree"]["tree_sha256"]
    snapshot = plan["source_snapshot"]
    assert snapshot["execution_sha"] == plan["runtime_lock"]["execution_sha"]
    assert snapshot["canonical_tree_sha256"] == snapshot["git_receipt"]["verification"]["mounted_tree_canonical_sha256"]
    assert snapshot["construction"] == "clean_git_archive"
    assert snapshot["read_only_required"] is True
    assert snapshot["worktree_mount_forbidden"] is True
    assert runner._tree_manifest(Path(snapshot["host_path"]))["tree_sha256"] == snapshot["tree"]["tree_sha256"]


def test_writable_output_roots_are_physically_and_lexically_disjoint(
    tmp_path: Path,
) -> None:
    plan, _, _ = _resolved_plan(tmp_path)
    repo = Path(plan["path_roots"]["repo"]["host"])

    nested = copy.deepcopy(plan)
    nested["path_roots"]["output"]["host"] = str(repo / "nested-output")
    nested["plan_sha256"] = runner._plan_hash(nested)
    with pytest.raises(ValueError, match="output host root overlaps"):
        runner.validate_execution_plan(nested)

    alias = tmp_path / "repo-alias"
    alias.symlink_to(repo, target_is_directory=True)
    aliased = copy.deepcopy(plan)
    aliased["path_roots"]["output"]["host"] = str(alias / "nested-output")
    aliased["plan_sha256"] = runner._plan_hash(aliased)
    with pytest.raises(ValueError, match="output host root overlaps"):
        runner.validate_execution_plan(aliased)

    container_nested = copy.deepcopy(plan)
    container_nested["path_roots"]["output"]["container"] = (
        plan["path_roots"]["repo"]["container"] + "/nested-output"
    )
    container_nested["plan_sha256"] = runner._plan_hash(container_nested)
    with pytest.raises(ValueError, match="output container root overlaps"):
        runner.validate_execution_plan(container_nested)


def test_finalized_nested_overlay_cannot_capture_e1_3_destinations(
    tmp_path: Path,
) -> None:
    plan, _, _ = _resolved_plan(tmp_path)
    output = Path(plan["path_roots"]["output"]["host"])
    finalized = output / "formal-stage"
    runner._validate_finalized_overlay(plan, finalized)

    colliding = copy.deepcopy(plan)
    next(
        shard for shard in colliding["shards"]
        if shard["stage"] == runner.STAGE_E1_3
    )["output_relative"] = "formal-stage/illegal-child"
    with pytest.raises(ValueError, match="falls inside finalized"):
        runner._validate_finalized_overlay(colliding, finalized)
    with pytest.raises(ValueError, match="may not contain"):
        runner._validate_finalized_overlay(plan, output)


@pytest.mark.parametrize(
    "image",
    [
        "sha256:" + "b" * 64,
        "registry.example/fpct:latest",
        "registry.example/fpct:tag@sha256:" + "b" * 64,
    ],
)
def test_execution_plan_rejects_unnamed_or_tagged_image_references(
    tmp_path: Path, image: str,
) -> None:
    plan, _, _ = _resolved_plan(tmp_path)
    changed = copy.deepcopy(plan)
    changed["runtime_lock"]["image_digest"] = image
    changed["plan_sha256"] = runner._plan_hash(changed)
    with pytest.raises(RuntimeError, match="execution SHA/image digest"):
        runner.validate_execution_plan(changed, for_execution=True)


def test_runtime_probe_receipt_evidence_is_exactly_cross_bound(
    tmp_path: Path,
) -> None:
    plan, plan_path, _ = _resolved_plan(tmp_path)
    runtime_record = plan["runtime_lock"]["runtime_provenance"]
    runtime_path = runner.resolve_logical_path(plan, runtime_record["logical_path"])
    original = runtime_path.read_bytes()
    payload = json.loads(original)
    payload["source_snapshot_verification"]["receipt_file_sha256"] = "0" * 64
    runtime_path.write_bytes(runner.canonical_json_bytes(payload))
    changed = copy.deepcopy(plan)
    changed_runtime = changed["runtime_lock"]["runtime_provenance"]
    changed_runtime["sha256"] = runner.sha256_file(runtime_path)
    changed_runtime["identity"]["source_snapshot_verification"] = payload[
        "source_snapshot_verification"
    ]
    changed["plan_sha256"] = runner._plan_hash(changed)
    plan_path.write_bytes(runner.canonical_json_bytes(changed))
    try:
        with pytest.raises(ValueError, match="runtime probe identity changed"):
            runner.verify_plan_sources(changed)
    finally:
        runtime_path.write_bytes(original)


def test_consolidated_gate_has_exact_check_set_and_exact_sha(tmp_path: Path) -> None:
    snapshot = _source_snapshot(tmp_path)
    gate = _gate(tmp_path, snapshot)
    resolver = lambda logical: snapshot / runner._parse_logical(logical)[1]
    record = runner.verify_instrumentation_gate(gate, runner.sha256_file(gate), evidence_resolver=resolver)
    assert record["checks"] == {name: True for name in runner.REQUIRED_GATE_CHECKS}
    with pytest.raises(RuntimeError, match="exact SHA"):
        runner.verify_instrumentation_gate(gate, "0" * 64, evidence_resolver=resolver)
    payload = json.loads(gate.read_text())
    payload["checks"].pop("formula_oracles")
    gate.write_bytes(runner.canonical_json_bytes(payload))
    with pytest.raises(RuntimeError, match="exact consolidated"):
        runner.verify_instrumentation_gate(gate, evidence_resolver=resolver)


def test_prepared_input_lock_membership_is_recomputed_not_format_checked(tmp_path: Path) -> None:
    plan, _, _ = _resolved_plan(tmp_path)
    manifest_path = runner.resolve_logical_path(plan, plan["input_lock"]["prepared"]["manifest"]["logical_path"])
    sidecar_path = runner.resolve_logical_path(plan, plan["input_lock"]["prepared"]["sidecar"]["logical_path"])
    payload = json.loads(manifest_path.read_text())
    payload["task_contract"]["openbookqa"]["membership_sha256"] = "f" * 64
    manifest_path.write_bytes(runner.canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="does not recompute"):
        runner._prepared_input_lock(
            manifest_path, sidecar_path, plan["group_contract"],
            repo_root=Path(plan["path_roots"]["repo"]["host"]),
        )


def test_raw_topology_lock_blocks_artifact_tamper_before_any_shard(tmp_path: Path) -> None:
    plan, _, _ = _resolved_plan(tmp_path)
    runner.verify_plan_sources(plan)
    artifact_name = next(iter(plan["raw_topology_lock"]["artifacts"]))
    artifact = runner.resolve_logical_path(plan, plan["raw_topology_lock"]["root"]) / artifact_name
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="raw-topology artifact provenance changed"):
        runner.verify_plan_sources(plan)


def test_unresolved_template_plan_fails_before_backend_import(tmp_path: Path) -> None:
    snapshot = _source_snapshot(tmp_path)
    receipt, payload = _snapshot_receipt(snapshot)
    plan = runner.build_execution_plan(
        snapshot, tmp_path / "missing", tmp_path / "out",
        require_checkpoints=False, execution_sha=payload["execution_sha"],
        source_snapshot_receipt=receipt,
    )
    path = tmp_path / "unresolved.json"
    path.write_bytes(runner.canonical_json_bytes(plan))
    called = False
    def loader(_):
        nonlocal called
        called = True
    with pytest.raises(RuntimeError, match="unresolved"):
        _execute_shard(path, plan["shards"][0]["shard_id"], None, tmp_path / "out", None, claim_id="claim", backend_loader=loader)
    assert called is False


def test_run_shard_rejects_replaced_plan_or_nonmechanical_claim_before_backend(
    tmp_path: Path,
) -> None:
    plan, plan_path, gate = _resolved_plan(tmp_path)
    shard = plan["shards"][0]
    backend_loaded = False

    def loader(_spec):
        nonlocal backend_loaded
        backend_loaded = True
        return pytest.fail("backend must not load")

    replacement = copy.deepcopy(plan)
    replacement["replacement_nonce"] = "self-consistent-but-not-rendered"
    replacement["plan_sha256"] = runner._plan_hash(replacement)
    replacement_path = tmp_path / "replacement-plan.json"
    replacement_path.write_bytes(runner.canonical_json_bytes(replacement))
    with pytest.raises(RuntimeError, match="differs from rendered expected SHA"):
        runner.execute_shard(
            replacement_path, shard["shard_id"], gate, tmp_path / "output",
            "fixture:backend", expected_plan_sha256=plan["plan_sha256"],
            claim_id=runner._expected_claim_id(
                plan["plan_sha256"], shard["shard_id"]
            ),
            backend_loader=loader,
        )
    with pytest.raises(RuntimeError, match="claim ID differs"):
        runner.execute_shard(
            plan_path, shard["shard_id"], gate, tmp_path / "output",
            "fixture:backend", expected_plan_sha256=plan["plan_sha256"],
            claim_id="0" * 64, backend_loader=loader,
        )
    assert backend_loaded is False


def test_run_shard_cli_requires_and_forwards_expected_plan_sha(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    expected = "a" * 64
    observed = {}

    def fake_execute(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {"status": "GO"}

    monkeypatch.setattr(runner, "execute_shard", fake_execute)
    arguments = [
        "run-shard", "--plan", str(tmp_path / "plan.json"),
        "--expected-plan-sha256", expected,
        "--shard-id", "synthetic-shard",
        "--output-root", str(tmp_path / "output"),
        "--claim-id", "b" * 64,
    ]
    assert runner.main(arguments) == 0
    assert observed["kwargs"]["expected_plan_sha256"] == expected
    assert observed["kwargs"]["claim_id"] == "b" * 64
    assert json.loads(capsys.readouterr().out)["status"] == "GO"

    missing = [
        "run-shard", "--plan", str(tmp_path / "plan.json"),
        "--shard-id", "synthetic-shard",
        "--output-root", str(tmp_path / "output"),
        "--claim-id", "b" * 64,
    ]
    with pytest.raises(SystemExit):
        runner.main(missing)


def test_render_k8s_cli_requires_and_forwards_mounted_receipts(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    observed = {}

    def fake_render(*args):
        observed["args"] = args
        return {"status": "GO"}

    monkeypatch.setattr(runner, "render_k8s", fake_render)
    paths = {
        "plan": tmp_path / "plan.json",
        "template": tmp_path / "template.yaml",
        "output": tmp_path / "rendered",
        "bundle": tmp_path / "bundle.json",
        "receipt": tmp_path / "bundle-verification.json",
        "finalized_bundle": tmp_path / "finalized-bundle.json",
        "finalized_receipt": tmp_path / "finalized-bundle-verification.json",
    }
    arguments = [
        "render-k8s",
        "--plan", str(paths["plan"]),
        "--template", str(paths["template"]),
        "--output-dir", str(paths["output"]),
        "--stage", runner.STAGE_E1_3,
        "--lock-bundle-manifest", str(paths["bundle"]),
        "--lock-bundle-verification-receipt", str(paths["receipt"]),
        "--finalized-lock-bundle-manifest", str(paths["finalized_bundle"]),
        "--finalized-lock-bundle-verification-receipt",
        str(paths["finalized_receipt"]),
    ]
    assert runner.main(arguments) == 0
    assert observed["args"] == (
        paths["plan"], paths["template"], paths["output"], runner.STAGE_E1_3,
        paths["bundle"], paths["receipt"], paths["finalized_bundle"],
        paths["finalized_receipt"],
    )
    assert json.loads(capsys.readouterr().out)["status"] == "GO"

    receipt_index = arguments.index("--lock-bundle-verification-receipt")
    missing_required = arguments[:receipt_index] + arguments[receipt_index + 2:]
    with pytest.raises(SystemExit):
        runner.main(missing_required)


def test_backend_cannot_supply_cpost_and_f_is_joined_from_baseline(tmp_path: Path, monkeypatch) -> None:
    plan, plan_path, gate = _resolved_plan(tmp_path)
    cpost = _find_shard(plan, seed=runner.SEEDS[0], arm="c_post_trained", task="openbookqa", value=0.0, stage=runner.STAGE_E1_2)
    factorized = _find_shard(plan, seed=runner.SEEDS[0], arm="c_post_trained", task="openbookqa", value=1.0, stage=runner.STAGE_E1_2)
    output = tmp_path / "outputs"
    baseline_result = _execute_shard(plan_path, cpost["shard_id"], gate, output, "fixture:backend", claim_id="baseline", backend_loader=lambda _: _backend_for(plan, cpost))
    assert baseline_result["schema_boundary"]["physical_chunk_rows"] == 4096
    assert baseline_result["parquet_chunk_count"] == runner.TASK_GROUP_COUNTS["openbookqa"]
    assert baseline_result["streaming_receipt"]["order"] == "sample_sha256,row_ordinal"
    monkeypatch.setattr(runner, "_verify_phase_marker", lambda *_args, **_kwargs: {"status": "GO"})
    result = _execute_shard(plan_path, factorized["shard_id"], gate, output, "fixture:backend", claim_id="factorized", backend_loader=lambda _: _backend_for(plan, factorized))
    assert result["row_key_attestation"]["exact_bijection_with_cpost"] is True
    first = _first_capture_row(output / factorized["output_relative"])
    assert first["cpost_gold_logp"] == -1.25
    assert first["gold_logp"] == pytest.approx(-1.24)
    assert first["cpost_end_task_correct"] is True
    group = next(item for item in plan["group_contract"]["openbookqa"] if item["content_group_sha256"] == first["content_group_sha256"])
    assert first["alignment_sha256"] != group["prompt_alignment_sha256"]
    assert "cpost_gold_logp" not in runner._audit_columns()[0]

    bad_output = tmp_path / "bad-output"
    with pytest.raises(ValueError, match="may not provide"):
        _execute_shard(plan_path, cpost["shard_id"], gate, bad_output, "fixture:backend", claim_id="bad", backend_loader=lambda _: _backend_for(plan, cpost, inject_cpost=True))


def test_exact_row_key_bijection_rejects_missing_factorized_row(tmp_path: Path, monkeypatch) -> None:
    plan, plan_path, gate = _resolved_plan(tmp_path)
    cpost = _find_shard(plan, seed=runner.SEEDS[0], arm="f_trained", task="openbookqa", value=0.0, stage=runner.STAGE_E1_2)
    factorized = _find_shard(plan, seed=runner.SEEDS[0], arm="f_trained", task="openbookqa", value=1.0, stage=runner.STAGE_E1_2)
    output = tmp_path / "outputs"
    _execute_shard(plan_path, cpost["shard_id"], gate, output, "fixture:backend", claim_id="c", backend_loader=lambda _: _backend_for(plan, cpost))
    monkeypatch.setattr(runner, "_verify_phase_marker", lambda *_args, **_kwargs: {"status": "GO"})
    with pytest.raises(ValueError, match="row-key universe"):
        _execute_shard(plan_path, factorized["shard_id"], gate, output, "fixture:backend", claim_id="f", backend_loader=lambda _: _backend_for(plan, factorized, drop_last=True))


def test_f_runtime_lambda_zero_is_executed_and_exactly_joins_cpost(tmp_path: Path, monkeypatch) -> None:
    plan, plan_path, gate = _resolved_plan(tmp_path)
    baseline = _find_shard(plan, seed=runner.SEEDS[0], arm="c_post_trained", task="openbookqa", value=0.0, stage=runner.STAGE_E1_2)
    control = _find_shard(plan, seed=runner.SEEDS[0], arm="c_post_trained", task="openbookqa", value=0.0, stage=runner.STAGE_E1_3)
    output = tmp_path / "outputs"
    _execute_shard(plan_path, baseline["shard_id"], gate, output, "fixture:backend", claim_id="baseline", backend_loader=lambda _: _backend_for(plan, baseline))
    monkeypatch.setattr(runner, "_verify_finalized_e1_2_lock", lambda *_args, **_kwargs: {"status": "GO"})
    result = _execute_shard(plan_path, control["shard_id"], gate, output, "fixture:backend", claim_id="f-lambda-zero", finalized_lock_sha256="f" * 64, backend_loader=lambda _: _backend_for(plan, control))
    assert result["row_key_attestation"]["exact_bijection_with_cpost"] is True
    row = _first_capture_row(output / control["output_relative"])
    assert row["inference_operator"] == "f"
    assert row["lambda_value"] == 0.0
    assert row["gamma"] == row["prior"]
    assert row["gold_logp"] == row["cpost_gold_logp"]
    assert row["end_task_correct"] == row["cpost_end_task_correct"]
    assert row["output_delta_l2"] == 0.0


def test_completed_shard_is_idempotent_without_backend_reload(tmp_path: Path) -> None:
    plan, plan_path, gate = _resolved_plan(tmp_path)
    shard = _find_shard(plan, seed=runner.SEEDS[0], arm="c_post_trained", task="openbookqa", value=0.0, stage=runner.STAGE_E1_2)
    output, calls = tmp_path / "outputs", 0
    def loader(_):
        nonlocal calls
        calls += 1
        return _backend_for(plan, shard)
    _execute_shard(plan_path, shard["shard_id"], gate, output, "fixture:backend", claim_id="first", backend_loader=loader)
    resumed = _execute_shard(plan_path, shard["shard_id"], gate, output, "fixture:backend", claim_id="first", backend_loader=loader)
    assert resumed["resumed_without_model_load"] is True
    assert calls == 1
    claim = json.loads(runner._claim_path(output, shard).read_text())
    assert claim["status"] == "SUCCEEDED" and claim["attempt"] == 1
    with pytest.raises(RuntimeError, match="claim ID differs"):
        runner.execute_shard(
            plan_path, shard["shard_id"], gate, output, "fixture:backend",
            expected_plan_sha256=plan["plan_sha256"], claim_id="other",
            backend_loader=loader,
        )


def test_e1_3_completed_resume_rechecks_every_lock_before_fast_return(
    tmp_path: Path, monkeypatch,
) -> None:
    plan, plan_path, gate = _resolved_plan(tmp_path)
    baseline = _find_shard(
        plan, seed=runner.SEEDS[0], arm="c_post_trained",
        task="openbookqa", value=0.0, stage=runner.STAGE_E1_2,
    )
    sweep = _find_shard(
        plan, seed=runner.SEEDS[0], arm="c_post_trained",
        task="openbookqa", value=0.25, stage=runner.STAGE_E1_3,
    )
    output = tmp_path / "outputs"
    _execute_shard(
        plan_path, baseline["shard_id"], gate, output, "fixture:backend",
        claim_id="baseline", backend_loader=lambda _: _backend_for(plan, baseline),
    )
    receipt = tmp_path / "finalized-receipt.json"
    receipt.write_bytes(b"sealed-finalized-receipt")
    receipt_sha = runner.sha256_file(receipt)
    phase_calls = 0

    def verify_finalized(
        _plan, _output_root, *, expected_lock_sha256=None,
        receipt_path=None, **_kwargs,
    ):
        nonlocal phase_calls
        phase_calls += 1
        if expected_lock_sha256 != receipt_sha:
            raise RuntimeError("finalized receipt SHA mismatch")
        if receipt_path != receipt or not receipt.is_file():
            raise RuntimeError("finalized receipt missing")
        return {"status": "GO"}

    monkeypatch.setattr(runner, "_verify_finalized_e1_2_lock", verify_finalized)
    backend_loads = 0

    def loader(_spec):
        nonlocal backend_loads
        backend_loads += 1
        return _backend_for(plan, sweep)

    _execute_shard(
        plan_path, sweep["shard_id"], gate, output, "fixture:backend",
        claim_id="sweep", finalized_lock_sha256=receipt_sha,
        finalized_receipt_path=receipt, backend_loader=loader,
    )
    resumed = _execute_shard(
        plan_path, sweep["shard_id"], gate, output, "fixture:backend",
        claim_id="sweep", finalized_lock_sha256=receipt_sha,
        finalized_receipt_path=receipt, backend_loader=loader,
    )
    assert resumed["resumed_without_model_load"] is True
    assert backend_loads == 1 and phase_calls == 2

    with pytest.raises(RuntimeError, match="lacks the rendered finalized-lock SHA"):
        _execute_shard(
            plan_path, sweep["shard_id"], gate, output, "fixture:backend",
            claim_id="sweep", finalized_receipt_path=receipt,
            backend_loader=lambda _spec: pytest.fail("backend loaded"),
        )
    with pytest.raises(RuntimeError, match="finalized receipt SHA mismatch"):
        _execute_shard(
            plan_path, sweep["shard_id"], gate, output, "fixture:backend",
            claim_id="sweep", finalized_lock_sha256="0" * 64,
            finalized_receipt_path=receipt,
            backend_loader=lambda _spec: pytest.fail("backend loaded"),
        )
    missing_receipt = tmp_path / "missing-finalized-receipt.json"
    with pytest.raises(RuntimeError, match="finalized receipt missing"):
        _execute_shard(
            plan_path, sweep["shard_id"], gate, output, "fixture:backend",
            claim_id="sweep", finalized_lock_sha256=receipt_sha,
            finalized_receipt_path=missing_receipt,
            backend_loader=lambda _spec: pytest.fail("backend loaded"),
        )

    source = runner.resolve_logical_path(
        plan, plan["runtime_lock"]["backend"]["source"]["logical_path"]
    )
    source_bytes = source.read_bytes()
    source.write_bytes(source_bytes + b"tamper")
    try:
        with pytest.raises(ValueError, match="snapshot|source"):
            _execute_shard(
                plan_path, sweep["shard_id"], gate, output,
                "fixture:backend", claim_id="sweep",
                finalized_lock_sha256=receipt_sha,
                finalized_receipt_path=receipt,
                backend_loader=lambda _spec: pytest.fail("backend loaded"),
            )
    finally:
        source.write_bytes(source_bytes)

    gate_bytes = gate.read_bytes()
    gate.write_bytes(gate_bytes + b" ")
    try:
        with pytest.raises(RuntimeError, match="gate exact SHA"):
            _execute_shard(
                plan_path, sweep["shard_id"], gate, output,
                "fixture:backend", claim_id="sweep",
                finalized_lock_sha256=receipt_sha,
                finalized_receipt_path=receipt,
                backend_loader=lambda _spec: pytest.fail("backend loaded"),
            )
    finally:
        gate.write_bytes(gate_bytes)

    baseline_root = output / baseline["output_relative"]
    baseline_manifest = json.loads((baseline_root / "capture_manifest.json").read_text())
    sample_manifest_path = baseline_root / baseline_manifest["sample_streams"][0]["manifest_relative_path"]
    sample_manifest = json.loads(sample_manifest_path.read_text())
    baseline_parquet = sample_manifest_path.parent / sample_manifest["chunks"][0]["relative_path"]
    baseline_parquet.write_bytes(baseline_parquet.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="capture provenance mismatch"):
        _execute_shard(
            plan_path, sweep["shard_id"], gate, output, "fixture:backend",
            claim_id="sweep", finalized_lock_sha256=receipt_sha,
            finalized_receipt_path=receipt,
            backend_loader=lambda _spec: pytest.fail("backend loaded"),
        )


def test_failed_claim_records_error_and_same_claim_id_can_resume(tmp_path: Path) -> None:
    plan, plan_path, gate = _resolved_plan(tmp_path)
    shard = _find_shard(plan, seed=runner.SEEDS[0], arm="c_post_trained", task="openbookqa", value=0.0, stage=runner.STAGE_E1_2)
    output = tmp_path / "outputs"

    def failed_backend(_request):
        raise ArithmeticError("synthetic backend failure")

    with pytest.raises(ArithmeticError, match="synthetic backend failure"):
        _execute_shard(plan_path, shard["shard_id"], gate, output, "fixture:backend", claim_id="stable-claim", backend_loader=lambda _: failed_backend)
    failed = json.loads(runner._claim_path(output, shard).read_text())
    assert failed["status"] == "FAILED"
    assert failed["error_class"] == "ArithmeticError"
    assert failed["attempt"] == 1
    result = _execute_shard(plan_path, shard["shard_id"], gate, output, "fixture:backend", claim_id="stable-claim", backend_loader=lambda _: _backend_for(plan, shard))
    assert result["status"] == "GO"
    succeeded = json.loads(runner._claim_path(output, shard).read_text())
    assert succeeded["status"] == "SUCCEEDED"
    assert succeeded["attempt"] == 2


def test_streaming_template_nested_chunk_and_orphan_fail_before_backend(
    tmp_path: Path,
) -> None:
    plan, plan_path, gate = _resolved_plan(tmp_path)
    shard = _find_shard(
        plan, seed=runner.SEEDS[0], arm="c_post_trained",
        task="openbookqa", value=0.0, stage=runner.STAGE_E1_2,
    )
    index_path = runner.resolve_logical_path(
        plan,
        plan["input_lock"]["prepared"]["streaming_contract"]
        ["streaming_template_lock"]["index"]["logical_path"],
    )
    index = json.loads(index_path.read_text())
    nested_manifest = index_path.parent / index["records"][0][
        "manifest_relative_path"
    ]
    nested = json.loads(nested_manifest.read_text())
    physical = nested_manifest.parent / nested["chunks"][0]["relative_path"]
    original = physical.read_bytes()
    backend_loaded = False

    def loader(_spec):
        nonlocal backend_loaded
        backend_loaded = True
        return pytest.fail("backend must not load before recursive template GO")

    physical.write_bytes(original + b"nested-tamper")
    with pytest.raises(ValueError, match="streaming-template physical chunk"):
        _execute_shard(
            plan_path, shard["shard_id"], gate, tmp_path / "outputs",
            "fixture:backend", claim_id="nested-tamper", backend_loader=loader,
        )
    assert backend_loaded is False
    physical.write_bytes(original)

    orphan = nested_manifest.parent / "unreferenced-template-artifact.bin"
    orphan.write_bytes(b"orphan")
    with pytest.raises(ValueError, match="corruption or orphans"):
        _execute_shard(
            plan_path, shard["shard_id"], gate, tmp_path / "outputs",
            "fixture:backend", claim_id="nested-orphan", backend_loader=loader,
        )
    assert backend_loaded is False


def test_stable_staging_preserves_verified_prefix_and_resumes_exact_suffix(
    tmp_path: Path,
) -> None:
    plan, plan_path, gate = _resolved_plan(tmp_path)
    shard = _find_shard(
        plan, seed=runner.SEEDS[0], arm="c_post_trained",
        task="openbookqa", value=0.0, stage=runner.STAGE_E1_2,
    )
    output = tmp_path / "outputs"
    first_sample_rows = 28 * 16

    def interrupted_backend(request):
        result = _backend_for(plan, shard)(request)
        source = result["rows"]

        def interrupted_rows():
            for index, row in enumerate(source):
                if index == first_sample_rows + 1:
                    raise RuntimeError("synthetic stream interruption")
                yield row

        return {**result, "rows": interrupted_rows()}

    with pytest.raises(RuntimeError, match="synthetic stream interruption"):
        _execute_shard(
            plan_path, shard["shard_id"], gate, output,
            "fixture:backend", claim_id="stable-resume",
            backend_loader=lambda _spec: interrupted_backend,
        )
    output_dir = output / shard["output_relative"]
    staging = runner._capture_staging_path(output_dir)
    state = runner._inspect_capture_staging(output_dir, plan, shard)
    assert staging.is_dir() and not output_dir.exists()
    assert state["cursor"]["completed_logical_rows"] == first_sample_rows
    assert state["cursor"]["next_row_ordinal"] == 0
    assert state["cursor"]["completed_sample_count"] == 1
    first_sample = sorted(state["expected_streams"])[0]
    first_chunk = staging / "capture_chunks" / first_sample / "chunk-000000.parquet"
    first_receipt = staging / "capture_chunks" / first_sample / runner._chunk_receipt_name(0)
    immutable_hashes = {
        "chunk": runner.sha256_file(first_chunk),
        "receipt": runner.sha256_file(first_receipt),
    }

    stable_attestation_path = staging / "backend_attestation.json"
    stable_attestation_bytes = stable_attestation_path.read_bytes()
    premodel_loads = 0

    def attestation_must_not_load(_spec):
        nonlocal premodel_loads
        premodel_loads += 1
        return pytest.fail("invalid stable attestation must precede backend load")

    tampered_cases = []
    membership = json.loads(stable_attestation_bytes)
    membership["attestation"]["membership_sha256"] = "0" * 64
    tampered_cases.append(membership)
    runtime_asset = json.loads(stable_attestation_bytes)
    runtime_asset["attestation"]["runtime_assets"]["receiver"][
        "tree_sha256"
    ] = "0" * 64
    tampered_cases.append(runtime_asset)
    projector = json.loads(stable_attestation_bytes)
    projector["attestation"]["projector_load"]["records"][0][
        "missing_keys"
    ] = ["tampered"]
    tampered_cases.append(projector)
    expected_rows = json.loads(stable_attestation_bytes)
    expected_rows["attestation"]["expected_long_form_rows"]["sum"] += 1
    tampered_cases.append(expected_rows)
    unknown_key = json.loads(stable_attestation_bytes)
    unknown_key["attestation"]["unknown_schema_extension"] = True
    tampered_cases.append(unknown_key)
    for tampered_attestation in tampered_cases:
        stable_attestation_path.write_bytes(
            runner.canonical_json_bytes(tampered_attestation)
        )
        with pytest.raises(ValueError, match="stable backend"):
            _execute_shard(
                plan_path, shard["shard_id"], gate, output,
                "fixture:backend", claim_id="stable-resume",
                backend_loader=attestation_must_not_load,
            )
    assert premodel_loads == 0
    stable_attestation_path.write_bytes(stable_attestation_bytes)

    original_chunk = first_chunk.read_bytes()
    first_chunk.write_bytes(original_chunk + b"corruption")
    backend_loads = 0

    def must_not_load(_spec):
        nonlocal backend_loads
        backend_loads += 1
        return pytest.fail("corrupt staging must fail before backend load")

    with pytest.raises((ValueError, OSError), match="capture staging|Parquet|parquet"):
        _execute_shard(
            plan_path, shard["shard_id"], gate, output,
            "fixture:backend", claim_id="stable-resume",
            backend_loader=must_not_load,
        )
    assert backend_loads == 0 and first_chunk.exists()
    first_chunk.write_bytes(original_chunk)

    orphan = first_chunk.parent / "orphan.bin"
    orphan.write_bytes(b"orphan")
    with pytest.raises(ValueError, match="corruption or orphans"):
        _execute_shard(
            plan_path, shard["shard_id"], gate, output,
            "fixture:backend", claim_id="stable-resume",
            backend_loader=must_not_load,
        )
    assert backend_loads == 0 and first_chunk.exists()
    orphan.unlink()

    observed_cursors = []

    def resumed_backend(request):
        observed_cursors.append(dict(request["resume_cursor"]))
        return _backend_for(plan, shard)(request)

    result = _execute_shard(
        plan_path, shard["shard_id"], gate, output,
        "fixture:backend", claim_id="stable-resume",
        backend_loader=lambda _spec: resumed_backend,
    )
    assert result["status"] == "GO"
    assert len(observed_cursors) == 1
    assert observed_cursors[0]["completed_logical_rows"] == first_sample_rows
    assert observed_cursors[0]["next_sample_sha256"] == sorted(
        state["expected_streams"]
    )[1]
    assert runner.sha256_file(
        output_dir / "capture_chunks" / first_sample / "chunk-000000.parquet"
    ) == immutable_hashes["chunk"]
    assert runner.sha256_file(
        output_dir / "capture_chunks" / first_sample
        / runner._chunk_receipt_name(0)
    ) == immutable_hashes["receipt"]
    assert not staging.exists()
    verified = runner.verify_capture_artifacts(output_dir, plan)
    assert verified["status"] == "GO"


def test_multichunk_sample_resumes_at_4096_and_matches_clean_semantic_stream(
    tmp_path: Path, monkeypatch,
) -> None:
    from script.analysis.fpct_e1_streaming_verify import (
        SampleRowStream,
        canonical_endpoint_id,
        semantic_row_bytes,
    )

    query_count = runner.PARQUET_BATCH_ROWS + 1
    item = {
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
                "query_position": index,
                "target_position": index + 1,
                "target_token_id": 1000 + index,
            }
            for index in range(query_count)
        ],
        "certified_parents": [{
            "parent_position": 7,
            "candidate_count": 2,
            "prior": [0.6, 0.4],
            "candidate_indices": [10, 11, -1, -1],
            "candidate_valid_mask": [True, True, False, False],
            "candidate_slot_weights": [0.6, 0.4, 0.0, 0.0],
            "topology": "competing_overlap",
            "statistical_weight": 1.0,
        }],
    }
    schema_sha = "a" * 64
    shard = {
        "shard_id": "synthetic-multichunk-shard",
        "seed": runner.SEEDS[0],
        "checkpoint_arm": "c_post_trained",
        "inference_operator": "c_post",
        "cell": "Y_CC",
        "task": "openbookqa",
        "lambda_value": 0.0,
        "checkpoint_tree_sha256": "b" * 64,
        "checkpoint_id": "synthetic-checkpoint",
        "membership_sha256": "c" * 64,
        "output_relative": "synthetic/multichunk",
    }
    runtime_stream = SampleRowStream(
        item, num_layers=1, num_query_heads=1, num_kv_heads=1,
        endpoint_id=canonical_endpoint_id(
            shard["seed"], shard["checkpoint_arm"],
            shard["inference_operator"], shard["cell"], shard["task"], 0.0,
        ),
        schema_sha256=schema_sha,
    )
    template_stream = SampleRowStream(
        item, num_layers=1, num_query_heads=1, num_kv_heads=1,
        endpoint_id=canonical_endpoint_id(
            0, "input_geometry", "input_geometry", "INPUT_LOCK",
            shard["task"], 0.0,
        ),
        schema_sha256=schema_sha,
    )
    template_digest = hashlib.sha256()
    for identity in template_stream.iter_rows():
        template_digest.update(semantic_row_bytes(identity))
    shard["expected_row_template"] = {
        "count": runtime_stream.row_count,
        "semantic_stream_sha256": template_digest.hexdigest(),
        "schema_sha256": schema_sha,
        "physical_chunk_rows": runner.PARQUET_BATCH_ROWS,
        "ordinal_order": "sample_sha256,row_ordinal",
    }
    plan = {
        "plan_sha256": "d" * 64,
        "runtime_lock": {
            "execution_sha": "e" * 40,
            "image_digest": FAKE_IMAGE,
        },
        "input_lock": {"prepared": {
            "streaming_contract": {"schema_sha256": schema_sha},
            "runtime_assets": {
                role: {
                    "model_id": role, "resolved_path": f"/models/{role}",
                    "tree_sha256": value, "file_count": 1, "bytes": 2,
                }
                for role, value in {"receiver": "7" * 64, "sender": "8" * 64}.items()
            },
            "task_contract": {"openbookqa": {
                "expected_long_form_rows": {
                    "count": 1, "sum": runtime_stream.row_count,
                },
            }},
        }},
        "group_contract": {"openbookqa": [{
            "content_group_sha256": item["content_group_sha256"],
            "sample_sha256": [item["sample_sha256"]],
        }]},
        "instrumentation_gate": {"status": "GO"},
        "checkpoints": [{
            "checkpoint_id": "synthetic-checkpoint",
            "final": {"files": [
                {"relative": "projector_0.json", "sha256": "9" * 64},
                {"relative": "projector_0.pt", "sha256": "f" * 64},
            ]},
        }],
        "shards": [shard],
    }
    monkeypatch.setattr(
        runner, "_capture_sample_streams",
        lambda *_args, **_kwargs: (
            {item["sample_sha256"]: runtime_stream},
            {item["sample_sha256"]: template_stream},
        ),
    )
    rows = [_row(shard, runtime_stream.row_at(index)) for index in range(runtime_stream.row_count)]
    claim = {"claim_id": "synthetic", "attempt": 1}

    def attestation(cursor):
        return {
            **_fixture_stable_attestation(plan, shard),
            "resume_cursor_sha256": cursor["cursor_sha256"],
            "resume_completed_logical_rows": cursor["completed_logical_rows"],
            "resume_next_sample_sha256": cursor["next_sample_sha256"],
            "resume_next_row_ordinal": cursor["next_row_ordinal"],
        }

    def prefix_channel(cursor, *, observed_sha=None):
        observed = (
            cursor["partial_sample_backend_prefix_sha256"]
            if observed_sha is None else observed_sha
        )
        return {
            "schema_version": 1,
            "protocol_id": runner.CAPTURE_STAGING_PROTOCOL_ID,
            "cursor_sha256": cursor["cursor_sha256"],
            "expected_row_count": cursor[
                "partial_sample_backend_prefix_row_count"
            ],
            "expected_backend_projection_sha256": cursor[
                "partial_sample_backend_prefix_sha256"
            ],
            "status": "GO_EXACT_PREFIX_MATCH",
            "observed_row_count": cursor[
                "partial_sample_backend_prefix_row_count"
            ],
            "observed_backend_projection_sha256": observed,
            "exact_match": observed
            == cursor["partial_sample_backend_prefix_sha256"],
        }

    interrupted_output = tmp_path / "interrupted"
    initial = runner._inspect_capture_staging(
        interrupted_output, plan, shard, create=True,
    )["cursor"]

    def interrupted_rows():
        for index, row in enumerate(rows):
            yield row
            if index == runner.PARQUET_BATCH_ROWS:
                raise RuntimeError("interrupt after chunk zero plus one row")

    with pytest.raises(RuntimeError, match="interrupt after chunk zero"):
        runner.write_capture_artifacts(
            interrupted_rows(), interrupted_output, shard, plan,
            attestation(initial), plan["instrumentation_gate"], claim, None,
            expected_resume_cursor=initial,
            resume_prefix_verification=prefix_channel(initial),
        )
    resumed = runner._inspect_capture_staging(
        interrupted_output, plan, shard,
    )
    assert resumed["cursor"]["next_sample_sha256"] == item["sample_sha256"]
    assert resumed["cursor"]["next_row_ordinal"] == runner.PARQUET_BATCH_ROWS
    assert resumed["cursor"]["completed_logical_rows"] == runner.PARQUET_BATCH_ROWS
    staged_chunk = (
        resumed["root"] / "capture_chunks" / item["sample_sha256"]
        / "chunk-000000.parquet"
    )
    chunk_zero_sha = runner.sha256_file(staged_chunk)
    with pytest.raises(ValueError, match="recomputed prefix differs"):
        runner.write_capture_artifacts(
            iter(rows[runner.PARQUET_BATCH_ROWS:]),
            interrupted_output, shard, plan, attestation(resumed["cursor"]),
            plan["instrumentation_gate"], {**claim, "attempt": 2}, None,
            expected_resume_cursor=resumed["cursor"],
            resume_prefix_verification=prefix_channel(
                resumed["cursor"], observed_sha="0" * 64,
            ),
        )
    assert runner.sha256_file(staged_chunk) == chunk_zero_sha
    resumed_manifest = runner.write_capture_artifacts(
        iter(rows[runner.PARQUET_BATCH_ROWS:]),
        interrupted_output, shard, plan, attestation(resumed["cursor"]),
        plan["instrumentation_gate"], {**claim, "attempt": 2}, None,
        expected_resume_cursor=resumed["cursor"],
        resume_prefix_verification=prefix_channel(resumed["cursor"]),
    )
    assert runner.sha256_file(
        interrupted_output / "capture_chunks" / item["sample_sha256"]
        / "chunk-000000.parquet"
    ) == chunk_zero_sha

    clean_output = tmp_path / "clean"
    clean_cursor = runner._inspect_capture_staging(
        clean_output, plan, shard, create=True,
    )["cursor"]
    clean_manifest = runner.write_capture_artifacts(
        iter(rows), clean_output, shard, plan, attestation(clean_cursor),
        plan["instrumentation_gate"], claim, None,
        expected_resume_cursor=clean_cursor,
        resume_prefix_verification=prefix_channel(clean_cursor),
    )
    assert (
        resumed_manifest["streaming_receipt"]["semantic_stream_sha256"]
        == clean_manifest["streaming_receipt"]["semantic_stream_sha256"]
    )
    assert runner.verify_capture_artifacts(interrupted_output, plan)["status"] == "GO"
    assert runner.verify_capture_artifacts(clean_output, plan)["status"] == "GO"

    orphan_output = tmp_path / "orphan-recovery"
    orphan_cursor = runner._inspect_capture_staging(
        orphan_output, plan, shard, create=True,
    )["cursor"]

    def fail_after_hardlink(stage, _record):
        if stage == "after_chunk_hardlink_before_receipt":
            raise RuntimeError("injected post-hardlink crash")

    with pytest.raises(RuntimeError, match="post-hardlink crash"):
        runner.write_capture_artifacts(
            iter(rows), orphan_output, shard, plan, attestation(orphan_cursor),
            plan["instrumentation_gate"], claim, None,
            expected_resume_cursor=orphan_cursor,
            resume_prefix_verification=prefix_channel(orphan_cursor),
            failure_injector=fail_after_hardlink,
        )
    orphan_staging = runner._capture_staging_path(orphan_output)
    uncommitted = (
        orphan_staging / "capture_chunks" / item["sample_sha256"]
        / "chunk-000000.parquet"
    )
    assert uncommitted.is_file()
    second_orphan = uncommitted.with_name("chunk-000001.parquet")
    second_orphan.write_bytes(uncommitted.read_bytes())
    orphan_attestation = orphan_staging / "backend_attestation.json"
    first_orphan_sha = runner.sha256_file(uncommitted)
    second_orphan_sha = runner.sha256_file(second_orphan)
    orphan_attestation_sha = runner.sha256_file(orphan_attestation)
    with pytest.raises(ValueError, match="chunk/receipt prefix"):
        runner._inspect_capture_staging(orphan_output, plan, shard)
    assert runner.sha256_file(uncommitted) == first_orphan_sha
    assert runner.sha256_file(second_orphan) == second_orphan_sha
    assert runner.sha256_file(orphan_attestation) == orphan_attestation_sha
    second_orphan.unlink()
    repaired = runner._inspect_capture_staging(
        orphan_output, plan, shard,
    )
    assert repaired["cursor"]["completed_logical_rows"] == 0
    assert repaired["cursor"]["next_row_ordinal"] == 0
    assert not uncommitted.exists()


@pytest.mark.parametrize(
    "relative",
    (
        "capture_staging_identity.json",
        "backend_attestation.json",
        "capture_chunks/sample/chunk-000000.receipt.json",
        "capture_chunks/sample/chunk_manifest.json",
        "capture_manifest.json",
    ),
)
def test_metadata_hardlink_crash_temp_never_enters_stable_tree(
    tmp_path: Path, relative: str,
) -> None:
    stable = tmp_path / "stable"
    working = tmp_path / "sibling-work"
    path = stable / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    working.mkdir()
    payload = runner.canonical_json_bytes({
        "artifact": relative, "immutable": True,
    })
    observed = []

    def crash(stage, record):
        assert stage == "after_metadata_hardlink_before_temp_cleanup"
        artifact = Path(record["artifact_path"])
        temporary = Path(record["temporary_path"])
        assert artifact == path
        assert temporary.is_file() and artifact.is_file()
        assert stable not in temporary.parents
        observed.append((artifact, temporary))
        raise RuntimeError("synthetic metadata post-hardlink crash")

    with pytest.raises(RuntimeError, match="metadata post-hardlink crash"):
        runner._write_once(
            path, payload, temporary_root=working, failure_injector=crash,
        )
    assert observed and path.read_bytes() == payload
    assert not any(
        child.name.startswith(".") or child.suffix == ".tmp"
        for child in stable.rglob("*")
    )
    runner._write_once(path, payload, temporary_root=working)
    assert path.read_bytes() == payload


def test_write_once_rejects_same_byte_symlink_destination(tmp_path: Path) -> None:
    stable = tmp_path / "stable"
    stable.mkdir()
    working = tmp_path / "work"
    working.mkdir()
    external = tmp_path / "external.json"
    payload = runner.canonical_json_bytes({"same": "bytes"})
    external.write_bytes(payload)
    destination = stable / "capture_manifest.json"
    destination.symlink_to(external)
    with pytest.raises(ValueError, match="not a regular file"):
        runner._write_once(destination, payload, temporary_root=working)
    assert destination.is_symlink() and external.read_bytes() == payload


def test_nonmechanical_claim_is_rejected_and_active_lease_is_exclusive(tmp_path: Path) -> None:
    plan, plan_path, gate = _resolved_plan(tmp_path)
    shard = _find_shard(plan, seed=runner.SEEDS[0], arm="c_post_trained", task="openbookqa", value=0.0, stage=runner.STAGE_E1_2)
    output = tmp_path / "outputs"
    with pytest.raises(LookupError):
        _execute_shard(plan_path, shard["shard_id"], gate, output, "fixture:backend", claim_id="owner-a", backend_loader=lambda _: (lambda _request: (_ for _ in ()).throw(LookupError("fail"))))
    with pytest.raises(RuntimeError, match="claim ID differs"):
        runner.execute_shard(
            plan_path, shard["shard_id"], gate, output, "fixture:backend",
            expected_plan_sha256=plan["plan_sha256"], claim_id="owner-b",
            backend_loader=lambda _: _backend_for(plan, shard),
        )

    second_output = tmp_path / "second-output"
    claim_path = runner._claim_path(second_output, shard)
    mechanical_claim = runner._expected_claim_id(
        plan["plan_sha256"], shard["shard_id"]
    )
    lease = runner._acquire_claim(claim_path, plan, shard, mechanical_claim)
    try:
        with pytest.raises(RuntimeError, match="currently active"):
            runner._acquire_claim(claim_path, plan, shard, mechanical_claim)
    finally:
        lease.transition("FAILED", error_class="SyntheticInterruption")
        lease.close()


def test_sweep_fails_before_backend_until_endpoint_completion_marker(tmp_path: Path) -> None:
    plan, plan_path, gate = _resolved_plan(tmp_path)
    shard = _find_shard(plan, seed=runner.SEEDS[0], arm="c_post_trained", task="openbookqa", value=0.25, stage=runner.STAGE_E1_3)
    called = False
    def loader(_):
        nonlocal called
        called = True
    with pytest.raises(RuntimeError, match="lacks the rendered finalized-lock SHA"):
        _execute_shard(plan_path, shard["shard_id"], gate, tmp_path / "outputs", "fixture:backend", claim_id="sweep", backend_loader=loader)
    assert called is False


def test_verify_all_closes_exactly_36_endpoints_and_full_grid(tmp_path: Path, monkeypatch) -> None:
    plan, plan_path, _ = _resolved_plan(tmp_path)
    def fake_closure(_plan, _output, stage, *, deep):
        del deep
        return [
            {
                "shard_id": shard["shard_id"],
                "lambda_value": shard["lambda_value"],
                "semantic_stream_sha256": _sha("semantic:" + shard["shard_id"]),
                "sample_manifest_tree_sha256": _sha("chunks:" + shard["shard_id"]),
                "parquet_chunk_count": 1,
                "row_key_sha256": _sha("keys:" + shard["shard_id"]),
                "row_count": 1,
                "capture_manifest_sha256": _sha("manifest:" + shard["shard_id"]),
            }
            for shard in runner._selected_shards(plan, stage)
        ]
    monkeypatch.setattr(runner, "_closure_records", fake_closure)
    monkeypatch.setattr(runner, "_verify_finalized_e1_2_lock", lambda *_args, **_kwargs: {"status": "GO"})
    output = tmp_path / "output"
    baselines = runner.verify_all(plan_path, output, runner.PHASE_E1_2_BASELINES, write_completion=True)
    factorized = runner.verify_all(plan_path, output, runner.PHASE_E1_2_FACTORIZED)
    endpoint = runner.verify_all(plan_path, output, runner.STAGE_E1_2, write_completion=True)
    full = runner.verify_all(plan_path, tmp_path / "output", runner.STAGE_E1_3)
    assert baselines["shard_count"] == 18
    assert factorized["shard_count"] == 18
    assert endpoint["shard_count"] == 36
    assert full["shard_count"] == 108
    assert endpoint["closure_sha256"] != full["closure_sha256"]


def test_finalize_stage_mechanically_merges_and_invokes_frozen_analyzer(tmp_path: Path, monkeypatch) -> None:
    plan, plan_path, _ = _resolved_plan(tmp_path)
    output = Path(plan["path_roots"]["output"]["host"])
    closure_sha = "a" * 64
    _write_phase_marker(plan, output, runner.STAGE_E1_2, closure_sha)
    calls = []

    def fake_verify_all(_plan_path, _output_root, stage, merge_dir=None, write_completion=False):
        assert stage == runner.STAGE_E1_2 and merge_dir is not None and write_completion is False
        merged = merge_dir / f"{stage}_rows.parquet"
        merged.write_bytes(b"synthetic-merged-parquet")
        return {"status": "GO", "stage": stage, "plan_sha256": plan["plan_sha256"], "closure_sha256": closure_sha, "deterministic_merge": {"path": str(merged), "sha256": runner.sha256_file(merged), "bytes": merged.stat().st_size, "row_count": plan["stages"][runner.STAGE_E1_2]["expected_total_rows"]}}

    import script.analysis.fpct_e1_mechanism_audit as analyzer
    monkeypatch.setattr(runner, "_verify_phase_marker", lambda *_args, **_kwargs: {"status": "GO", "closure_sha256": closure_sha})
    monkeypatch.setattr(runner, "verify_all", fake_verify_all)
    monkeypatch.setattr(analyzer, "read_input_rows", lambda path: calls.append(("read", path.name)) or [{"synthetic": True}])
    def fake_write(rows, directory):
        calls.append(("write", len(rows)))
        for filename in analyzer.OUTPUT_NAMES.values():
            (directory / filename).write_bytes(("synthetic:" + filename).encode())
        return {"status": "GO"}
    monkeypatch.setattr(analyzer, "write_artifacts", fake_write)
    monkeypatch.setattr(analyzer, "verify_artifacts", lambda directory: calls.append(("verify", directory.name)) or {"status": "GO", "row_count": 1, "artifact_sha256": {filename: runner.sha256_file(directory / filename) for filename in analyzer.OUTPUT_NAMES.values()}, "e1_pilot_consumed": False, "confirmatory_consumed": False})
    final_dir = output / "formal-stage"
    result = runner.finalize_stage(plan_path, output, runner.STAGE_E1_2, final_dir)
    assert result["status"] == "GO"
    assert result["closure_sha256"] == closure_sha
    assert (final_dir / "stage_artifact_manifest.json").is_file()
    frozen = json.loads((final_dir / "stage_artifact_manifest.json").read_text())
    assert frozen["merged_capture"]["relative_path"].endswith("_rows.parquet")
    assert "path" not in frozen["merged_capture"]
    assert runner._verify_finalized_e1_2_lock(plan, output)["status"] == "GO"
    assert [name for name, _ in calls] == ["read", "write", "verify"]
    merged = final_dir / frozen["merged_capture"]["relative_path"]
    merged.write_bytes(merged.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="merged capture changed"):
        runner.finalize_stage(plan_path, output, runner.STAGE_E1_2, final_dir)


def test_mounted_byte_receipt_binds_bundle_manifest_and_observed_configmaps(
    tmp_path: Path,
) -> None:
    bundle_manifest = {
        "schema_version": runner.K8S_LOCK_SCHEMA_VERSION,
        "protocol_id": runner.K8S_LOCK_PROTOCOL_ID,
        "status": "FROZEN_NOT_APPLIED",
        "execution_sha": "a" * 40,
        "plan_sha256": "b" * 64,
        "configmaps": {
            "plan_gate": {
                "name": "fpct-e1-plan-gate",
                "keys": {
                    "plan.json": {"bytes": 11, "sha256": "c" * 64},
                    "gate.json": {"bytes": 13, "sha256": "d" * 64},
                },
            },
            "runtime_probe": {
                "name": "fpct-e1-runtime-probe",
                "keys": {
                    "runtime.json": {"bytes": 17, "sha256": "e" * 64},
                },
            },
        },
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_bytes(runner.canonical_json_bytes(bundle_manifest))
    receipt = {
        "schema_version": runner.K8S_LOCK_SCHEMA_VERSION,
        "protocol_id": runner.K8S_LOCK_PROTOCOL_ID,
        "status": "VERIFIED_MOUNTED_KEY_BYTES",
        "execution_sha": bundle_manifest["execution_sha"],
        "plan_sha256": bundle_manifest["plan_sha256"],
        "bundle_manifest_sha256": runner.sha256_file(bundle_path),
        "configmaps": runner._expected_mounted_configmaps(bundle_manifest),
        "network_accessed": False,
        "kubectl_invoked_by_verifier": False,
    }
    receipt_path = tmp_path / "mounted-byte-receipt.json"
    receipt_path.write_bytes(runner.canonical_json_bytes(receipt))
    verified = runner._verify_mounted_bundle_receipt(
        bundle_manifest_path=bundle_path,
        bundle_manifest=bundle_manifest,
        receipt_path=receipt_path,
        finalized=False,
    )
    assert verified["status"] == "VERIFIED_MOUNTED_KEY_BYTES"
    assert verified["bundle_manifest_sha256"] == runner.sha256_file(bundle_path)

    tampered = copy.deepcopy(receipt)
    tampered["configmaps"]["fpct-e1-plan-gate"]["keys"]["plan.json"]["bytes"] += 1
    receipt_path.write_bytes(runner.canonical_json_bytes(tampered))
    with pytest.raises(RuntimeError, match="absent/invalid"):
        runner._verify_mounted_bundle_receipt(
            bundle_manifest_path=bundle_path,
            bundle_manifest=bundle_manifest,
            receipt_path=receipt_path,
            finalized=False,
        )


def test_finalized_mounted_byte_receipt_binds_closure_and_artifact_tree(
    tmp_path: Path,
) -> None:
    bundle_manifest = {
        "schema_version": runner.K8S_LOCK_SCHEMA_VERSION,
        "protocol_id": runner.K8S_FINALIZED_LOCK_PROTOCOL_ID,
        "status": "FROZEN_FINALIZED_NOT_APPLIED",
        "execution_sha": "a" * 40,
        "plan_sha256": "b" * 64,
        "receipt_sha256": "c" * 64,
        "closure_sha256": "d" * 64,
        "artifact_tree_sha256": "e" * 64,
        "configmaps": {
            "finalized": {
                "name": "fpct-e1-finalized",
                "keys": {
                    "e1_2_finalized_receipt.json": {
                        "bytes": 19,
                        "sha256": "f" * 64,
                    },
                },
            },
        },
    }
    bundle_path = tmp_path / "finalized-bundle.json"
    bundle_path.write_bytes(runner.canonical_json_bytes(bundle_manifest))
    receipt = {
        "schema_version": runner.K8S_LOCK_SCHEMA_VERSION,
        "protocol_id": runner.K8S_FINALIZED_LOCK_PROTOCOL_ID,
        "status": "VERIFIED_FINALIZED_MOUNTED_KEY_BYTES",
        "execution_sha": bundle_manifest["execution_sha"],
        "plan_sha256": bundle_manifest["plan_sha256"],
        "bundle_manifest_sha256": runner.sha256_file(bundle_path),
        "receipt_sha256": bundle_manifest["receipt_sha256"],
        "closure_sha256": bundle_manifest["closure_sha256"],
        "artifact_tree_sha256": bundle_manifest["artifact_tree_sha256"],
        "configmaps": runner._expected_mounted_configmaps(bundle_manifest),
        "network_accessed": False,
        "kubectl_invoked_by_verifier": False,
    }
    receipt_path = tmp_path / "finalized-mounted-byte-receipt.json"
    receipt_path.write_bytes(runner.canonical_json_bytes(receipt))
    verified = runner._verify_mounted_bundle_receipt(
        bundle_manifest_path=bundle_path,
        bundle_manifest=bundle_manifest,
        receipt_path=receipt_path,
        finalized=True,
    )
    assert verified["status"] == "VERIFIED_FINALIZED_MOUNTED_KEY_BYTES"

    tampered = copy.deepcopy(receipt)
    tampered["closure_sha256"] = "0" * 64
    receipt_path.write_bytes(runner.canonical_json_bytes(tampered))
    with pytest.raises(RuntimeError, match="finalized.*identity"):
        runner._verify_mounted_bundle_receipt(
            bundle_manifest_path=bundle_path,
            bundle_manifest=bundle_manifest,
            receipt_path=receipt_path,
            finalized=True,
        )


def test_render_k8s_is_separate_and_removes_every_placeholder(tmp_path: Path, monkeypatch) -> None:
    plan, plan_path, _ = _resolved_plan(tmp_path)
    template = runner.resolve_logical_path(plan, plan["k8s_template"]["logical_path"])
    fake_bundle = {"configmaps": {"plan_gate": {"name": "plan-gate"}, "runtime_probe": {"name": "runtime-probe"}}}
    monkeypatch.setattr(runner, "_verify_lock_bundle_manifest", lambda *_args, **_kwargs: fake_bundle)
    monkeypatch.setattr(
        runner,
        "_verify_mounted_bundle_receipt",
        lambda *, finalized, **_kwargs: {
            "status": (
                "VERIFIED_FINALIZED_MOUNTED_KEY_BYTES"
                if finalized else "VERIFIED_MOUNTED_KEY_BYTES"
            ),
            "sha256": ("f" if finalized else "e") * 64,
            "bytes": 123,
            "bundle_manifest_sha256": ("d" if finalized else "c") * 64,
            "configmaps": {},
        },
    )
    monkeypatch.setattr(runner, "_verify_finalized_e1_2_lock", lambda *_args, **_kwargs: {"status": "GO", "output_dir": "output://formal-stage"})
    monkeypatch.setattr(runner, "_verify_finalized_bundle_manifest", lambda *_args, **_kwargs: {"configmaps": {"finalized": {"name": "finalized-lock"}}})
    lock_path = Path(plan["path_roots"]["output"]["host"]) / "locks" / f"{runner.FINALIZED_E1_2}.json"
    runner.atomic_write(lock_path, runner.canonical_json_bytes({"synthetic": True}))
    result = runner.render_k8s(
        plan_path, template, tmp_path / "rendered", runner.STAGE_E1_3,
        tmp_path / "bundle.json", tmp_path / "bundle-verification.json",
        tmp_path / "finalized-bundle.json",
        tmp_path / "finalized-bundle-verification.json",
    )
    assert len(result["jobs"]) == 72
    assert result["requires_phase"] == runner.FINALIZED_E1_2
    assert result["resources"] == {
        "node_name": "4090-48gx2",
        "gpu_per_job": 1,
        "maximum_concurrent_jobs_on_node": 2,
        "scheduling": "remaining jobs stay Pending; no preemption",
    }
    assert result["run_identity"]["execution_prefix"] == plan["runtime_lock"]["execution_sha"][:8]
    assert result["mounted_byte_verification"]["initial"]["status"] == "VERIFIED_MOUNTED_KEY_BYTES"
    assert result["mounted_byte_verification"]["finalized"]["status"] == "VERIFIED_FINALIZED_MOUNTED_KEY_BYTES"
    text = (tmp_path / "rendered" / result["jobs"][0]["path"]).read_text()
    assert runner.UNRESOLVED.search(text) is None
    assert f'requires-phase: "{runner.FINALIZED_E1_2}"' in text
    assert "e1-pilot-access: \"false\"" in text
    assert 'nvidia.com/gpu: "1"' in text
    assert "PYTHONDONTWRITEBYTECODE" in text
    assert "C2C_MODEL_ROOT" in text
    assert "nodeName: 4090-48gx2" in text
    assert f"name: fpct-e1-{plan['runtime_lock']['execution_sha'][:8]}-" in text
    assert str(plan["source_snapshot"]["host_path"]) in text
    job = yaml.safe_load(text)
    assert job["metadata"]["annotations"][
        "fpct.openai.com/expected-plan-sha256"
    ] == plan["plan_sha256"]
    pod_spec = job["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    assert container["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
    }
    arguments = container["args"]
    expected_index = arguments.index("--expected-plan-sha256")
    assert arguments[expected_index + 1] == plan["plan_sha256"]
    claim_index = arguments.index("--claim-id")
    expected_claim = runner._expected_claim_id(
        plan["plan_sha256"], result["jobs"][0]["shard_id"]
    )
    assert arguments[claim_index + 1] == expected_claim
    environment = {record["name"]: record["value"] for record in container["env"]}
    assert {
        "HOME": "/tmp/home",
        "XDG_CACHE_HOME": "/tmp/xdg-cache",
        "HF_HOME": "/tmp/hf-home",
        "HF_DATASETS_CACHE": "/tmp/hf-datasets-cache",
        "TORCH_HOME": "/tmp/torch-home",
        "CUDA_CACHE_PATH": "/tmp/cuda-cache",
        "WANDB_DIR": "/tmp/wandb",
        "TMPDIR": "/tmp",
    }.items() <= environment.items()
    mounts = {record["name"]: record for record in container["volumeMounts"]}
    assert mounts["tmp"] == {"name": "tmp", "mountPath": "/tmp"}
    assert mounts["e1-results"].get("readOnly") is not True
    assert all(
        record.get("readOnly") is True
        for name, record in mounts.items()
        if name not in {"tmp", "e1-results"}
    )
    volumes = {record["name"]: record for record in pod_spec["volumes"]}
    assert volumes["tmp"] == {"name": "tmp", "emptyDir": {}}
    assert set(volumes["e1-results"]) == {"name", "hostPath"}


def test_k8s_endpoint_render_is_baseline_marker_then_factorized(tmp_path: Path, monkeypatch) -> None:
    plan, plan_path, _ = _resolved_plan(tmp_path)
    template = runner.resolve_logical_path(plan, plan["k8s_template"]["logical_path"])
    fake_bundle = {"configmaps": {"plan_gate": {"name": "plan-gate"}, "runtime_probe": {"name": "runtime-probe"}}}
    monkeypatch.setattr(runner, "_verify_lock_bundle_manifest", lambda *_args, **_kwargs: fake_bundle)
    monkeypatch.setattr(
        runner,
        "_verify_mounted_bundle_receipt",
        lambda **_kwargs: {
            "status": "VERIFIED_MOUNTED_KEY_BYTES",
            "sha256": "e" * 64,
            "bytes": 123,
            "bundle_manifest_sha256": "c" * 64,
            "configmaps": {},
        },
    )
    baseline = runner.render_k8s(plan_path, template, tmp_path / "baseline-jobs", runner.PHASE_E1_2_BASELINES, tmp_path / "bundle.json", tmp_path / "bundle-verification.json")
    assert len(baseline["jobs"]) == 18 and baseline["requires_phase"] is None
    with pytest.raises(FileNotFoundError):
        runner.render_k8s(plan_path, template, tmp_path / "factor-jobs-blocked", runner.PHASE_E1_2_FACTORIZED, tmp_path / "bundle.json", tmp_path / "bundle-verification.json")
    monkeypatch.setattr(runner, "_verify_phase_marker", lambda *_args, **_kwargs: {"status": "GO"})
    factorized = runner.render_k8s(plan_path, template, tmp_path / "factor-jobs", runner.PHASE_E1_2_FACTORIZED, tmp_path / "bundle.json", tmp_path / "bundle-verification.json")
    assert len(factorized["jobs"]) == 18
    assert factorized["requires_phase"] == runner.PHASE_E1_2_BASELINES
    with pytest.raises(ValueError, match="dependency-safe phase"):
        runner.render_k8s(plan_path, template, tmp_path / "unsafe-36", runner.STAGE_E1_2, tmp_path / "bundle.json", tmp_path / "bundle-verification.json")


def test_teacher_forced_helper_uses_t_plus_one_and_explicit_capture_lifecycle() -> None:
    labels = torch.tensor([[-100, -100, 3, 4, -100]])
    logits = torch.full((1, 5, 8), -30.0)
    logits[0, 1, 3] = 30.0; logits[0, 2, 4] = 30.0
    class Model:
        @staticmethod
        def fpct_teacher_forced_query_mask(value):
            mask = torch.zeros_like(value, dtype=torch.bool); mask[:, :-1] = value[:, 1:] != -100; return mask
        def begin_fpct_capture(self, **kwargs): self.begin = kwargs
        def __call__(self, **kwargs): return SimpleNamespace(logits=logits, loss=torch.tensor(0.25))
        @staticmethod
        def end_fpct_capture(): return {"schema_version": 1, "stores_raw_kv": False, "layers": {}}
    model = Model()
    observation = runner.teacher_forced_capture(model, {"input_ids": torch.zeros_like(labels)}, labels, metadata={"sample": "x"})
    assert model.begin["query_mask"].tolist() == [[False, True, True, False, False]]
    assert observation.gold["query_position"].tolist() == [1, 2]
    assert observation.gold["target_position"].tolist() == [2, 3]
    assert torch.all(observation.gold["gold_logp"] > -1e-6)


def test_capture_adapter_rejects_summary_only_and_raw_kv() -> None:
    with pytest.raises(RuntimeError, match="contract version"):
        list(runner.capture_report_long_rows({"stores_raw_kv": False, "layers": {}}))
    with pytest.raises(RuntimeError, match="stores_raw_kv"):
        list(runner.capture_report_long_rows({"stores_raw_kv": True, "long_form_primitives": []}))
    primitives = [{"layer": 0}]
    assert list(runner.capture_report_long_rows({"stores_raw_kv": False, "long_form_contract_version": 1, "long_form_incomplete_chunk_count": 0, "long_form_primitives": primitives})) == primitives
    with pytest.raises(RuntimeError, match="incomplete"):
        list(runner.capture_report_long_rows({"stores_raw_kv": False, "long_form_contract_version": 1, "long_form_incomplete_chunk_count": 1, "long_form_primitives": primitives}))
