from __future__ import annotations

import ast
import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from script.analysis import fpct_e1_a5_prompt_gate as gate
from script.analysis.fpct_e1_streaming_synthetic_gate import (
    BUFFER_COPY_MULTIPLIER,
    PHYSICAL_CHUNK_ROWS,
    STREAMING_SCHEMA,
    _canonical_full_schema_row_bound,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64


def _stream_case(*, rows: int, peak_rss: int) -> dict:
    return {
        "logical_rows": rows,
        "emitted_rows": rows,
        "semantic_stream_sha256": SHA,
        "replay_semantic_stream_sha256": SHA,
        "chunk_count": 1,
        "physical_file_bytes": 1,
        "max_physical_chunk_bytes": 1,
        "max_observed_canonical_row_bytes": 1,
        "peak_rss_bytes": peak_rss,
    }


def _streaming_gate_value() -> dict:
    canonical_bound = _canonical_full_schema_row_bound(gate.sha256_file(STREAMING_SCHEMA))
    allowance = BUFFER_COPY_MULTIPLIER * PHYSICAL_CHUNK_ROWS * canonical_bound
    baseline_rss = 100_000_000
    threshold = baseline_rss + allowance
    return {
        "estimated_physical_bytes_per_row": 1
        << (canonical_bound - 1).bit_length(),
        "streaming_stress": {
            "baseline": _stream_case(rows=8192, peak_rss=baseline_rss),
            "stress": _stream_case(rows=1_000_001, peak_rss=threshold),
            "threshold_bytes": threshold,
            "bounded_peak_rss": True,
        },
        "streaming_contract_checks": {
            name: True for name in gate.STREAMING_CONTRACT_CHECK_NAMES
        },
    }


def _schema_gate_value() -> dict:
    checks = {
        "a4_streaming_oracles_pass": True,
        "renderer_source_identity_attested": True,
        "a5_instrumentation_reattestation_go": True,
        "dual_anchor_tests_pass": True,
        "full_population_fixture_tests_pass": True,
        "prompt_classifier_tests_pass": True,
        "streaming_semantic_replay_tests_pass": True,
        "failure_corruption_atomicity_resume_tests_pass": True,
        "four_choice_exact_pass": True,
        "five_choice_gold_A_extra_E_pass": True,
        "five_choice_gold_E_fail_closed_pass": True,
        "first_four_order_change_fail_closed_pass": True,
        "suffix_E_text_change_detected_pass": True,
        "renderer_mismatch_fail_closed_pass": True,
        "chunk_partition_semantic_equivalence_pass": True,
    }
    value = {
        "schema_version": 7,
        "protocol_id": gate.A5_PROTOCOL_ID,
        "artifact_type": "a5_prompt_synthetic_gate",
        "status": "GO_PRE_NATURAL_SYNTHETIC_HARD_GATE",
        "execution_tree_sha256": SHA,
        "evidence_sha256": SHA,
        "tracked_files_sha256": {"safe/path": SHA},
        **_streaming_gate_value(),
        "environment": {
            "python": "3.10",
            "python_executable": "/python",
            "platform": "linux",
            "machine": "x86_64",
            "pyarrow": "1",
            "torch": "1",
            "transformers": "1",
            "tokenizers": "1",
            "datasets": "1",
            "pyyaml": "1",
            "cuda_visible_devices": "",
            "offline": True,
            "model_instantiated": False,
        },
        "environment_identity_sha256": SHA,
        "historical_a4_gate_sha256": gate.HISTORICAL_A4_GATE_SHA256,
        "historical_a4_gate_verified_against_current_tree": False,
        "tests": {"passed": 1, "failed": 0, "output_sha256": SHA},
        "checks": checks,
        "natural_e0_design_accessed": False,
        "model_instantiated": False,
        "model_or_checkpoint_loaded": False,
        "model_forward_run": False,
        "gpu_or_kubernetes_used": False,
        "e1_pilot_consumed": False,
    }
    return value


def test_a5_gate_schema_is_strict_for_firewall_streaming_and_tracked_files() -> None:
    value = _schema_gate_value()
    gate.validate_a5_schema_artifact(value)
    corruptions = []
    missing_streaming = copy.deepcopy(value)
    del missing_streaming["streaming_stress"]
    corruptions.append(missing_streaming)
    model_true = copy.deepcopy(value)
    model_true["model_instantiated"] = True
    corruptions.append(model_true)
    environment_online = copy.deepcopy(value)
    environment_online["environment"]["offline"] = False
    corruptions.append(environment_online)
    unsafe_tracked_path = copy.deepcopy(value)
    unsafe_tracked_path["tracked_files_sha256"] = {"../escape": SHA}
    corruptions.append(unsafe_tracked_path)
    missing_stress_field = copy.deepcopy(value)
    del missing_stress_field["streaming_stress"]["stress"]["peak_rss_bytes"]
    corruptions.append(missing_stress_field)
    extra = copy.deepcopy(value)
    extra["unregistered"] = True
    corruptions.append(extra)
    for corrupted in corruptions:
        with pytest.raises(ValueError):
            gate.validate_a5_schema_artifact(corrupted)


def test_streaming_evidence_recomputes_frozen_threshold_and_semantics() -> None:
    value = _streaming_gate_value()
    gate._verify_streaming_evidence(value, REPO_ROOT)
    cases = []
    wrong_threshold = copy.deepcopy(value)
    wrong_threshold["streaming_stress"]["threshold_bytes"] += 1
    cases.append(wrong_threshold)
    semantic_tamper = copy.deepcopy(value)
    semantic_tamper["streaming_stress"]["stress"][
        "replay_semantic_stream_sha256"
    ] = "b" * 64
    cases.append(semantic_tamper)
    too_few_rows = copy.deepcopy(value)
    too_few_rows["streaming_stress"]["stress"]["logical_rows"] = 999_999
    too_few_rows["streaming_stress"]["stress"]["emitted_rows"] = 999_999
    cases.append(too_few_rows)
    rss_over = copy.deepcopy(value)
    rss_over["streaming_stress"]["stress"]["peak_rss_bytes"] += 1
    cases.append(rss_over)
    for corrupted in cases:
        with pytest.raises(ValueError):
            gate._verify_streaming_evidence(corrupted, REPO_ROOT)


def test_offline_cpu_environment_is_mechanically_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in gate._OFFLINE_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    gate._require_offline_cpu_environment()
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    with pytest.raises(RuntimeError, match="offline environment"):
        gate._require_offline_cpu_environment()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        gate._require_offline_cpu_environment()


def test_gate_publication_is_no_overwrite_and_leaves_no_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "validate_a5_schema_artifact", lambda *_args, **_kwargs: None)
    output = tmp_path / "gate.json"
    value = {"evidence": "synthetic"}
    gate.publish_gate(value, output, repo_root=tmp_path)
    expected = gate.canonical_json_bytes(value) + b"\n"
    assert output.read_bytes() == expected
    with pytest.raises(FileExistsError, match="immutable A5 gate"):
        gate.publish_gate(value, output, repo_root=tmp_path)
    assert output.read_bytes() == expected
    assert not list(tmp_path.glob(".gate.json.*.tmp"))


def test_gate_tracks_all_three_instrumentation_artifacts() -> None:
    tracked = set(gate.TRACKED_SOURCE_FILES)
    assert {
        relative.as_posix() for relative in gate.A5_INSTRUMENTATION_RELATIVES
    }.issubset(tracked)


def test_gate_tracks_source_snapshot_producer_and_test() -> None:
    tracked = set(gate.TRACKED_SOURCE_FILES)
    assert {
        "script/experiment/fpct_e1_source_snapshot_lock.py",
        "test/test_fpct_e1_source_snapshot_lock.py",
    }.issubset(tracked)


def test_gate_reattests_renderer_and_all_36_e0_configs() -> None:
    evidence = gate._attest_renderer_and_all_e0_configs(REPO_ROOT)
    assert evidence["renderer"]["renderer_source_identity_attested"] is True
    assert evidence["configs"]["evaluation_config_count"] == 36


@pytest.mark.parametrize("corruption", ["config_index", "rendered_config"])
def test_all_config_attestation_rejects_index_or_config_tamper(
    tmp_path: Path, corruption: str
) -> None:
    from script.experiment.fpct_e1_prepare_input_lock import (
        A5_PROMPT_CONTRACT_RELATIVE,
        _load_a5_prompt_contract,
        _verify_all_e0_prompt_configs,
    )

    contract_relatives = (
        A5_PROMPT_CONTRACT_RELATIVE,
        gate.A5_CONTRACT_RELATIVE,
    )
    index_relative = Path("recipe/eval_recipe/fpct_e0/rendered/config_index.json")
    for relative in (*contract_relatives, index_relative):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)
    index = json.loads((tmp_path / index_relative).read_text())
    evaluation_files = [
        str(record["filename"])
        for record in index["records"]
        if record.get("kind") == "evaluation"
    ]
    for filename in evaluation_files:
        source = REPO_ROOT / index_relative.parent / filename
        destination = tmp_path / index_relative.parent / filename
        shutil.copy2(source, destination)
    contract = _load_a5_prompt_contract(tmp_path)
    assert _verify_all_e0_prompt_configs(tmp_path, contract)[
        "evaluation_config_count"
    ] == 36
    if corruption == "config_index":
        path = tmp_path / index_relative
    else:
        path = tmp_path / index_relative.parent / evaluation_files[0]
    path.write_bytes(path.read_bytes() + b"\n# tamper\n")
    with pytest.raises(ValueError):
        _verify_all_e0_prompt_configs(tmp_path, contract)


def test_stress_threshold_is_frozen_before_stress_worker() -> None:
    tree = ast.parse(
        (REPO_ROOT / "script/analysis/fpct_e1_a5_prompt_gate.py").read_text()
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_streaming_oracles"
    )
    assignments: dict[str, int] = {}
    stress_line = None
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.lineno
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_fresh_worker"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "a5_million_row"
        ):
            stress_line = node.lineno
    assert stress_line is not None
    assert assignments["canonical_bound"] < stress_line
    assert assignments["allowance"] < stress_line
    assert assignments["threshold"] < stress_line


def test_instrumentation_sha_binding_cannot_collapse_to_one_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def verified(**kwargs):
        calls.append(kwargs)
        return {"status": "GO"}

    from script.analysis import fpct_e1_instrumentation_gate as instrumentation

    monkeypatch.setattr(instrumentation, "verify_a5_successor_attestation", verified)
    for index, relative in enumerate(gate.A5_INSTRUMENTATION_RELATIVES):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact-{index}\n".encode())
    bindings = gate._verify_instrumentation(tmp_path)
    assert len(calls) == 1
    assert set(bindings) == {
        relative.as_posix() for relative in gate.A5_INSTRUMENTATION_RELATIVES
    }
    assert len(set(bindings.values())) == 3
    for relative, digest in bindings.items():
        assert digest == hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
