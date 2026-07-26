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

import script.experiment.fpct_e1_capture_runner as runner


REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_IMAGE = "sha256:" + "b" * 64


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


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
                (directory / "projector_config.json").write_text('{"version":1}\n')
                (directory / "projector_0.pt").write_bytes(b"synthetic-projector")
            (attempt / trained / "checkpoint-64" / "training_state.pt").write_bytes(b"state")
    return root


def _source_snapshot(tmp_path: Path) -> Path:
    repository = tmp_path / "source-repository"
    snapshot = tmp_path / "source-snapshot"
    relative_files = {
        runner.SPLIT_MANIFEST_RELATIVE,
        runner.E1_MANIFEST_RELATIVE,
        runner.MECHANISM_SCHEMA_RELATIVE,
        runner.E0_DEV_MANIFEST_RELATIVE,
        runner.E0_CONFIG_INDEX_RELATIVE,
        runner.EXECUTOR_RELATIVE,
        runner.ANALYZER_RELATIVE,
        Path("script/experiment/fpct_e1_prepare_input_lock.py"),
        Path("script/experiment/fpct_e1_runtime_backend.py"),
        Path("script/experiment/fpct_e1_runtime_probe.py"),
        Path("script/experiment/fpct_e1_source_snapshot_lock.py"),
        Path("script/experiment/fpct_e1_k8s_lock_bundle.py"),
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
    }
    relative_files.update(path.relative_to(REPO_ROOT) for path in (REPO_ROOT / runner.E0_CONFIG_ROOT_RELATIVE).glob("*"))
    for relative in sorted(relative_files):
        source, destination = REPO_ROOT / relative, repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
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
    runtime = tmp_path / "runtime.json"
    runtime.write_bytes(runner.canonical_json_bytes({
        "schema_version": 1,
        "protocol_id": "fpct_e1_runtime_probe_v1",
        "status": "FROZEN_MODEL_OUTPUT_FREE_RUNTIME_PROBE",
        "execution_sha": execution_sha,
        "image_digest": FAKE_IMAGE,
        "source_snapshot_tree_sha": receipt_payload["snapshot"]["mounted_tree_canonical_sha256"],
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
    sidecar = tmp_path / "e1_input_lock.pt"
    dimensions = {"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8}
    items = []
    for task in runner.TASKS:
        for group in draft["group_contract"][task]:
            item = {
                "task": task,
                "sample_sha256": group["sample_sha256"][0],
                "content_group_sha256": group["content_group_sha256"],
                "descriptor": {
                    "rendered_prompt_sha256": group["rendered_prompt_sha256"],
                    "prompt_alignment_sha256": group["prompt_alignment_sha256"],
                },
                "rendered_prompt_sha256": group["rendered_prompt_sha256"],
                "expected_long_form_rows": 1,
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
    topology_item["item_semantic_sha256"] = runner._nested_semantic_sha256(
        {key: value for key, value in topology_item.items() if key != "item_semantic_sha256"}
    )
    torch.save({
        "schema_version": 1,
        "protocol_id": "fpct_e1_e0_design_input_lock_v1",
        "split_role": runner.ALLOWED_SPLIT_ROLE,
        "items": items,
        "dimensions": dimensions,
        "e1_pilot_consumed": False,
        "model_or_checkpoint_loaded": False,
        "cuda_initialized": False,
    }, sidecar)
    task_contract = {}
    for task in runner.TASKS:
        shard = next(row for row in draft["shards"] if row["stage"] == runner.STAGE_E1_2 and row["task"] == task and row["lambda_value"] == 0.0)
        rows = [_row(shard, group) for group in draft["group_contract"][task]]
        keys = sorted(runner._key_bytes(row, runner.ROW_TEMPLATE_COLUMNS) for row in rows)
        digest = hashlib.sha256()
        for key in keys:
            digest.update(key + b"\n")
        members = sorted((item for item in items if item["task"] == task), key=lambda item: item["sample_sha256"])
        task_contract[task] = {
            "group_count": len(draft["group_contract"][task]),
            "members": [{"sample_sha256": item["sample_sha256"], "content_group_sha256": item["content_group_sha256"], "item_semantic_sha256": item["item_semantic_sha256"]} for item in members],
            "membership_sha256": runner._nested_semantic_sha256(sorted([item["sample_sha256"], item["content_group_sha256"], item["item_semantic_sha256"]] for item in members)),
            "row_template": {"count": len(keys), "sha256": digest.hexdigest()},
            "expected_long_form_rows": runner._expected_long_form_rows_contract(members),
        }
    input_manifest = tmp_path / "e1_input_lock_manifest.json"
    input_manifest.write_bytes(runner.canonical_json_bytes({
        "schema_version": 1,
        "protocol_id": "fpct_e1_e0_design_input_lock_v1",
        "status": "FROZEN_CPU_INPUTS_NO_MODEL_OUTPUT",
        "split_role": runner.ALLOWED_SPLIT_ROLE,
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


def _write_phase_marker(plan: dict, output: Path, stage: str, closure: str = "4" * 64) -> Path:
    count = {runner.PHASE_E1_2_BASELINES: 18, runner.PHASE_E1_2_FACTORIZED: 18, runner.STAGE_E1_2: 36, runner.STAGE_E1_3: 108}[stage]
    marker = {"schema_version": 1, "status": "GO", "stage": stage, "plan_sha256": plan["plan_sha256"], "shard_count": count, "closure_sha256": closure}
    path = runner._phase_marker(output, stage)
    runner.atomic_write(path, runner.canonical_json_bytes(marker))
    return path


def _row(shard: dict, group: dict, query: int = 4) -> dict:
    value = shard["lambda_value"]
    prior = [0.6, 0.4]
    gamma = prior if value == 0 else [0.5 + 0.1 * min(value, 2), 0.5 - 0.1 * min(value, 2)]
    return {
        "schema_version": 1,
        "split_role": runner.ALLOWED_SPLIT_ROLE,
        "seed": shard["seed"],
        "checkpoint_arm": shard["checkpoint_arm"],
        "inference_operator": shard["inference_operator"],
        "cell": shard["cell"],
        "task": shard["task"],
        "sample_sha256": group["sample_sha256"][0],
        "content_group_sha256": group["content_group_sha256"],
        "input_sha256": _sha("input:" + group["content_group_sha256"]),
        "alignment_sha256": _sha("full-response-alignment:" + group["content_group_sha256"]),
        "labels_sha256": _sha("labels:" + group["content_group_sha256"]),
        "gold_response_sha256": _sha("gold:" + group["content_group_sha256"]),
        "layer": 0,
        "query_head": 0,
        "kv_head": 0,
        "query_position": query,
        "target_position": query + 1,
        "target_token_id": 7,
        "parent_position": 2,
        "candidate_count": 2,
        "topology": "partition_compositional",
        "lambda_value": value,
        "prior": prior,
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
    rows = [_row(shard, group) for group in plan["group_contract"][shard["task"]]]
    final_columns = runner._audit_columns()[1]
    keys = runner._row_key_columns(final_columns)
    return sorted(rows, key=lambda row: runner._key_bytes(row, keys))


def _first_parquet_row(path: Path) -> dict:
    import pyarrow.parquet as pq

    batch = next(pq.ParquetFile(path).iter_batches(batch_size=1))
    return batch.to_pylist()[0]


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
        if inject_cpost:
            rows[0]["cpost_gold_logp"] = 999.0
        if drop_last:
            rows = rows[:-1]
        return {
            "contract_version": runner.CAPTURE_BACKEND_CONTRACT_VERSION,
            "rows": iter(rows),
            "attestation": {
                "split_role": runner.ALLOWED_SPLIT_ROLE,
                "capture_mode": "teacher_forced_response",
                "causal_shift_verified": True,
                "stores_raw_kv": False,
                "plan_sha256": plan["plan_sha256"],
                "shard_id": shard["shard_id"],
                "execution_sha": plan["runtime_lock"]["execution_sha"],
                "image_digest": FAKE_IMAGE,
                "checkpoint_tree_sha256": shard["checkpoint_tree_sha256"],
                "membership_sha256": shard["membership_sha256"],
            },
        }
    return backend


def test_plan_freezes_two_stage_36_plus_72_graph_and_lambda_ids(tmp_path: Path) -> None:
    plan, _, _ = _resolved_plan(tmp_path)
    runner.validate_execution_plan(plan, require_checkpoints=True, for_execution=True)
    assert plan["shard_count"] == runner.EXPECTED_SHARD_COUNT == 108
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
    ):
        assert f"repo://{relative}" in frozen_sources


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
        runner._prepared_input_lock(manifest_path, sidecar_path, plan["group_contract"])


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
        runner.execute_shard(path, plan["shards"][0]["shard_id"], None, tmp_path / "out", None, claim_id="claim", backend_loader=loader)
    assert called is False


def test_backend_cannot_supply_cpost_and_f_is_joined_from_baseline(tmp_path: Path, monkeypatch) -> None:
    plan, plan_path, gate = _resolved_plan(tmp_path)
    monkeypatch.setattr(runner, "PARQUET_BATCH_ROWS", 16)
    cpost = _find_shard(plan, seed=runner.SEEDS[0], arm="c_post_trained", task="openbookqa", value=0.0, stage=runner.STAGE_E1_2)
    factorized = _find_shard(plan, seed=runner.SEEDS[0], arm="c_post_trained", task="openbookqa", value=1.0, stage=runner.STAGE_E1_2)
    output = tmp_path / "outputs"
    baseline_result = runner.execute_shard(plan_path, cpost["shard_id"], gate, output, "fixture:backend", claim_id="baseline", backend_loader=lambda _: _backend_for(plan, cpost))
    assert baseline_result["schema_boundary"]["parquet_batch_rows"] == 16
    assert baseline_result["parquet_row_group_count"] == 5
    import pyarrow.parquet as pq
    assert pq.ParquetFile(output / cpost["output_relative"] / "e1_capture_rows.parquet").num_row_groups == 5
    monkeypatch.setattr(runner, "_verify_phase_marker", lambda *_args, **_kwargs: {"status": "GO"})
    result = runner.execute_shard(plan_path, factorized["shard_id"], gate, output, "fixture:backend", claim_id="factorized", backend_loader=lambda _: _backend_for(plan, factorized))
    assert result["row_key_attestation"]["exact_bijection_with_cpost"] is True
    first = _first_parquet_row(output / factorized["output_relative"] / "e1_capture_rows.parquet")
    assert first["cpost_gold_logp"] == -1.25
    assert first["gold_logp"] == pytest.approx(-1.24)
    assert first["cpost_end_task_correct"] is True
    group = next(item for item in plan["group_contract"]["openbookqa"] if item["content_group_sha256"] == first["content_group_sha256"])
    assert first["alignment_sha256"] != group["prompt_alignment_sha256"]
    assert "cpost_gold_logp" not in runner._audit_columns()[0]

    bad_output = tmp_path / "bad-output"
    with pytest.raises(ValueError, match="may not provide"):
        runner.execute_shard(plan_path, cpost["shard_id"], gate, bad_output, "fixture:backend", claim_id="bad", backend_loader=lambda _: _backend_for(plan, cpost, inject_cpost=True))


def test_exact_row_key_bijection_rejects_missing_factorized_row(tmp_path: Path, monkeypatch) -> None:
    plan, plan_path, gate = _resolved_plan(tmp_path)
    cpost = _find_shard(plan, seed=runner.SEEDS[0], arm="f_trained", task="openbookqa", value=0.0, stage=runner.STAGE_E1_2)
    factorized = _find_shard(plan, seed=runner.SEEDS[0], arm="f_trained", task="openbookqa", value=1.0, stage=runner.STAGE_E1_2)
    output = tmp_path / "outputs"
    runner.execute_shard(plan_path, cpost["shard_id"], gate, output, "fixture:backend", claim_id="c", backend_loader=lambda _: _backend_for(plan, cpost))
    monkeypatch.setattr(runner, "_verify_phase_marker", lambda *_args, **_kwargs: {"status": "GO"})
    with pytest.raises(ValueError, match="row-key universe"):
        runner.execute_shard(plan_path, factorized["shard_id"], gate, output, "fixture:backend", claim_id="f", backend_loader=lambda _: _backend_for(plan, factorized, drop_last=True))


def test_f_runtime_lambda_zero_is_executed_and_exactly_joins_cpost(tmp_path: Path, monkeypatch) -> None:
    plan, plan_path, gate = _resolved_plan(tmp_path)
    baseline = _find_shard(plan, seed=runner.SEEDS[0], arm="c_post_trained", task="openbookqa", value=0.0, stage=runner.STAGE_E1_2)
    control = _find_shard(plan, seed=runner.SEEDS[0], arm="c_post_trained", task="openbookqa", value=0.0, stage=runner.STAGE_E1_3)
    output = tmp_path / "outputs"
    runner.execute_shard(plan_path, baseline["shard_id"], gate, output, "fixture:backend", claim_id="baseline", backend_loader=lambda _: _backend_for(plan, baseline))
    monkeypatch.setattr(runner, "_verify_finalized_e1_2_lock", lambda *_args, **_kwargs: {"status": "GO"})
    result = runner.execute_shard(plan_path, control["shard_id"], gate, output, "fixture:backend", claim_id="f-lambda-zero", finalized_lock_sha256="f" * 64, backend_loader=lambda _: _backend_for(plan, control))
    assert result["row_key_attestation"]["exact_bijection_with_cpost"] is True
    row = _first_parquet_row(output / control["output_relative"] / "e1_capture_rows.parquet")
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
    runner.execute_shard(plan_path, shard["shard_id"], gate, output, "fixture:backend", claim_id="first", backend_loader=loader)
    resumed = runner.execute_shard(plan_path, shard["shard_id"], gate, output, "fixture:backend", claim_id="first", backend_loader=loader)
    assert resumed["resumed_without_model_load"] is True
    assert calls == 1
    claim = json.loads(runner._claim_path(output, shard).read_text())
    assert claim["status"] == "SUCCEEDED" and claim["attempt"] == 1
    with pytest.raises(RuntimeError, match="completed shard claim belongs"):
        runner.execute_shard(plan_path, shard["shard_id"], gate, output, "fixture:backend", claim_id="other", backend_loader=loader)


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
    runner.execute_shard(
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

    runner.execute_shard(
        plan_path, sweep["shard_id"], gate, output, "fixture:backend",
        claim_id="sweep", finalized_lock_sha256=receipt_sha,
        finalized_receipt_path=receipt, backend_loader=loader,
    )
    resumed = runner.execute_shard(
        plan_path, sweep["shard_id"], gate, output, "fixture:backend",
        claim_id="sweep", finalized_lock_sha256=receipt_sha,
        finalized_receipt_path=receipt, backend_loader=loader,
    )
    assert resumed["resumed_without_model_load"] is True
    assert backend_loads == 1 and phase_calls == 2

    with pytest.raises(RuntimeError, match="lacks the rendered finalized-lock SHA"):
        runner.execute_shard(
            plan_path, sweep["shard_id"], gate, output, "fixture:backend",
            claim_id="sweep", finalized_receipt_path=receipt,
            backend_loader=lambda _spec: pytest.fail("backend loaded"),
        )
    with pytest.raises(RuntimeError, match="finalized receipt SHA mismatch"):
        runner.execute_shard(
            plan_path, sweep["shard_id"], gate, output, "fixture:backend",
            claim_id="sweep", finalized_lock_sha256="0" * 64,
            finalized_receipt_path=receipt,
            backend_loader=lambda _spec: pytest.fail("backend loaded"),
        )
    missing_receipt = tmp_path / "missing-finalized-receipt.json"
    with pytest.raises(RuntimeError, match="finalized receipt missing"):
        runner.execute_shard(
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
            runner.execute_shard(
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
            runner.execute_shard(
                plan_path, sweep["shard_id"], gate, output,
                "fixture:backend", claim_id="sweep",
                finalized_lock_sha256=receipt_sha,
                finalized_receipt_path=receipt,
                backend_loader=lambda _spec: pytest.fail("backend loaded"),
            )
    finally:
        gate.write_bytes(gate_bytes)

    baseline_parquet = output / baseline["output_relative"] / "e1_capture_rows.parquet"
    baseline_parquet.write_bytes(baseline_parquet.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="capture provenance mismatch"):
        runner.execute_shard(
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
        runner.execute_shard(plan_path, shard["shard_id"], gate, output, "fixture:backend", claim_id="stable-claim", backend_loader=lambda _: failed_backend)
    failed = json.loads(runner._claim_path(output, shard).read_text())
    assert failed["status"] == "FAILED"
    assert failed["error_class"] == "ArithmeticError"
    assert failed["attempt"] == 1
    result = runner.execute_shard(plan_path, shard["shard_id"], gate, output, "fixture:backend", claim_id="stable-claim", backend_loader=lambda _: _backend_for(plan, shard))
    assert result["status"] == "GO"
    succeeded = json.loads(runner._claim_path(output, shard).read_text())
    assert succeeded["status"] == "SUCCEEDED"
    assert succeeded["attempt"] == 2


def test_failed_claim_rejects_different_claim_id_and_active_lease_is_exclusive(tmp_path: Path) -> None:
    plan, plan_path, gate = _resolved_plan(tmp_path)
    shard = _find_shard(plan, seed=runner.SEEDS[0], arm="c_post_trained", task="openbookqa", value=0.0, stage=runner.STAGE_E1_2)
    output = tmp_path / "outputs"
    with pytest.raises(LookupError):
        runner.execute_shard(plan_path, shard["shard_id"], gate, output, "fixture:backend", claim_id="owner-a", backend_loader=lambda _: (lambda _request: (_ for _ in ()).throw(LookupError("fail"))))
    with pytest.raises(RuntimeError, match="another plan/shard/claim-id"):
        runner.execute_shard(plan_path, shard["shard_id"], gate, output, "fixture:backend", claim_id="owner-b", backend_loader=lambda _: _backend_for(plan, shard))

    second_output = tmp_path / "second-output"
    claim_path = runner._claim_path(second_output, shard)
    lease = runner._acquire_claim(claim_path, plan, shard, "same-owner")
    try:
        with pytest.raises(RuntimeError, match="currently active"):
            runner._acquire_claim(claim_path, plan, shard, "same-owner")
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
        runner.execute_shard(plan_path, shard["shard_id"], gate, tmp_path / "outputs", "fixture:backend", claim_id="sweep", backend_loader=loader)
    assert called is False


def test_verify_all_closes_exactly_36_endpoints_and_full_grid(tmp_path: Path, monkeypatch) -> None:
    plan, plan_path, _ = _resolved_plan(tmp_path)
    def fake_closure(_plan, _output, stage, *, deep):
        del deep
        return [
            {
                "shard_id": shard["shard_id"],
                "lambda_value": shard["lambda_value"],
                "parquet_sha256": _sha("parquet:" + shard["shard_id"]),
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


def test_render_k8s_is_separate_and_removes_every_placeholder(tmp_path: Path, monkeypatch) -> None:
    plan, plan_path, _ = _resolved_plan(tmp_path)
    template = runner.resolve_logical_path(plan, plan["k8s_template"]["logical_path"])
    fake_bundle = {"configmaps": {"plan_gate": {"name": "plan-gate"}, "runtime_probe": {"name": "runtime-probe"}}}
    monkeypatch.setattr(runner, "_verify_lock_bundle_manifest", lambda *_args, **_kwargs: fake_bundle)
    monkeypatch.setattr(runner, "_verify_finalized_e1_2_lock", lambda *_args, **_kwargs: {"status": "GO", "output_dir": "output://formal-stage"})
    monkeypatch.setattr(runner, "_verify_finalized_bundle_manifest", lambda *_args, **_kwargs: {"configmaps": {"finalized": {"name": "finalized-lock"}}})
    lock_path = Path(plan["path_roots"]["output"]["host"]) / "locks" / f"{runner.FINALIZED_E1_2}.json"
    runner.atomic_write(lock_path, runner.canonical_json_bytes({"synthetic": True}))
    result = runner.render_k8s(plan_path, template, tmp_path / "rendered", runner.STAGE_E1_3, tmp_path / "bundle.json", tmp_path / "finalized-bundle.json")
    assert len(result["jobs"]) == 72
    assert result["requires_phase"] == runner.FINALIZED_E1_2
    assert result["resources"] == {
        "node_name": "4090-48gx2",
        "gpu_per_job": 1,
        "maximum_concurrent_jobs_on_node": 2,
        "scheduling": "remaining jobs stay Pending; no preemption",
    }
    assert result["run_identity"]["execution_prefix"] == plan["runtime_lock"]["execution_sha"][:8]
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


def test_k8s_endpoint_render_is_baseline_marker_then_factorized(tmp_path: Path, monkeypatch) -> None:
    plan, plan_path, _ = _resolved_plan(tmp_path)
    template = runner.resolve_logical_path(plan, plan["k8s_template"]["logical_path"])
    fake_bundle = {"configmaps": {"plan_gate": {"name": "plan-gate"}, "runtime_probe": {"name": "runtime-probe"}}}
    monkeypatch.setattr(runner, "_verify_lock_bundle_manifest", lambda *_args, **_kwargs: fake_bundle)
    baseline = runner.render_k8s(plan_path, template, tmp_path / "baseline-jobs", runner.PHASE_E1_2_BASELINES, tmp_path / "bundle.json")
    assert len(baseline["jobs"]) == 18 and baseline["requires_phase"] is None
    with pytest.raises(FileNotFoundError):
        runner.render_k8s(plan_path, template, tmp_path / "factor-jobs-blocked", runner.PHASE_E1_2_FACTORIZED, tmp_path / "bundle.json")
    monkeypatch.setattr(runner, "_verify_phase_marker", lambda *_args, **_kwargs: {"status": "GO"})
    factorized = runner.render_k8s(plan_path, template, tmp_path / "factor-jobs", runner.PHASE_E1_2_FACTORIZED, tmp_path / "bundle.json")
    assert len(factorized["jobs"]) == 18
    assert factorized["requires_phase"] == runner.PHASE_E1_2_BASELINES
    with pytest.raises(ValueError, match="dependency-safe phase"):
        runner.render_k8s(plan_path, template, tmp_path / "unsafe-36", runner.STAGE_E1_2, tmp_path / "bundle.json")


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
