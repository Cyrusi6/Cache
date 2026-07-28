#!/usr/bin/env python3
"""Build and verify the versioned FPCT-E1 A5 pre-natural hard gate.

The historical A4 receipt is hashed as predecessor evidence only.  It is never
re-verified against the changed A5 tree.  Gate publication is transactional and
contains no natural E0-design input or scientific output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from script.experiment.fpct_e1_a5_prompt_provenance import (
    A5_PROTOCOL_ID,
    attest_e0_renderer_identity,
    canonical_json_bytes,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
A5_SCHEMA_RELATIVE = Path("recipe/eval_recipe/fpct_e1/e1_a5_prompt_schema.json")
A5_CONTRACT_RELATIVE = Path("recipe/eval_recipe/fpct_e1/e1_a5_prompt_contract.json")
A5_GATE_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5_prompt_synthetic_gate.json"
)
HISTORICAL_A4_GATE_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_streaming_synthetic_gate.json"
)
HISTORICAL_A4_GATE_SHA256 = (
    "42b6c98fd5f27a49e258bae4b79ff3c4673c9144465a3c43f9a672d228c82df4"
)
A5_INSTRUMENTATION_RELATIVES = (
    Path("recipe/eval_recipe/fpct_e1/e1_instrumentation_parity_a5.json"),
    Path("recipe/eval_recipe/fpct_e1/e1_synthetic_query_variance_a5.json"),
    Path("recipe/eval_recipe/fpct_e1/e1_instrumentation_hard_gate_a5.json"),
)
TEST_FILES = (
    "test/test_fpct_e1_a5_prompt_gate.py",
    "test/test_fpct_e1_a5_prompt_provenance.py",
    "test/test_fpct_e1_a5_instrumentation_static.py",
    "test/test_fpct_e1_instrumentation_gate_a5.py",
    "test/test_fpct_e1_prepare_input_lock.py",
    "test/test_fpct_e1_streaming.py",
    "test/test_fpct_e1_streaming_synthetic_gate.py",
    "test/test_fpct_sealed_import.py",
    "test/test_fpct_e1_source_snapshot_lock.py",
)
TRACKED_SOURCE_FILES = tuple(
    sorted(
        {
            "script/analysis/fpct_e1_a5_prompt_gate.py",
            "script/analysis/fpct_e1_streaming_verify.py",
            "script/analysis/fpct_e1_streaming_synthetic_gate.py",
            "script/analysis/fpct_e1_instrumentation_gate.py",
            "script/experiment/fpct_e1_a5_prompt_provenance.py",
            "script/experiment/fpct_e1_prepare_input_lock.py",
            "script/experiment/fpct_e1_source_snapshot_lock.py",
            "script/experiment/fpct_e1_runtime_backend.py",
            "script/experiment/fpct_e1_capture_runner.py",
            "script/runtime/fpct_bootstrap.py",
            "rosetta/model/fpct_attention.py",
            "rosetta/model/fpct_instrumentation.py",
            "script/analysis/fpct_reference_operator.py",
            "FPCT_E1_A5_PRODUCTION_PROMPT_AMENDMENT.md",
            A5_SCHEMA_RELATIVE.as_posix(),
            A5_CONTRACT_RELATIVE.as_posix(),
            *(relative.as_posix() for relative in A5_INSTRUMENTATION_RELATIVES),
            *TEST_FILES,
        }
    )
)
_PASSED_PATTERN = re.compile(r"(?P<count>[0-9]+) passed")
_OFFLINE_ENVIRONMENT = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)
STREAMING_CONTRACT_CHECK_NAMES = (
    "row_key_reference_equivalence",
    "weights_reference_equivalence",
    "topology_reference_equivalence",
    "chunk_partition_semantic_equivalence",
    "aggregate_partition_equivalence",
    "bounded_peak_rss",
    "atomic_no_overwrite",
    "crash_resume_equivalence",
)


def validate_a5_schema_artifact(
    value: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> None:
    from script.analysis.fpct_e1_streaming_verify import (
        _load_manifest,
        _validate_json_schema,
    )

    schema = _load_manifest(repo_root / A5_SCHEMA_RELATIVE)
    _validate_json_schema(dict(value), schema, schema, path="$")


def _execution_tree_sha256(repo_root: Path) -> str:
    records = _tracked_source_sha256(repo_root)
    return hashlib.sha256(canonical_json_bytes(records)).hexdigest()


def _tracked_source_sha256(repo_root: Path) -> dict[str, str]:
    return {
        relative: sha256_file(repo_root / relative)
        for relative in TRACKED_SOURCE_FILES
    }


def _evidence_sha256(value: Mapping[str, Any]) -> str:
    evidence = {
        key: child for key, child in value.items() if key != "evidence_sha256"
    }
    return hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()


def _verify_instrumentation(repo_root: Path) -> dict[str, str]:
    from script.analysis.fpct_e1_instrumentation_gate import (
        verify_a5_successor_attestation,
    )

    parity_path, synthetic_path, hard_gate_path = (
        repo_root / relative for relative in A5_INSTRUMENTATION_RELATIVES
    )
    verify_a5_successor_attestation(
        repo_root=repo_root,
        parity_output=parity_path,
        synthetic_output=synthetic_path,
        hard_gate_output=hard_gate_path,
    )
    return {
        relative.as_posix(): sha256_file(repo_root / relative)
        for relative in A5_INSTRUMENTATION_RELATIVES
    }


def _require_offline_cpu_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("A5 synthetic gate requires CUDA_VISIBLE_DEVICES='' ")
    invalid = [name for name in _OFFLINE_ENVIRONMENT if os.environ.get(name) != "1"]
    if invalid:
        raise RuntimeError(
            "A5 synthetic gate requires offline environment variables: "
            + ", ".join(invalid)
        )


def _current_environment() -> dict[str, Any]:
    import pyarrow
    import torch

    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pyarrow": pyarrow.__version__,
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
        "datasets": importlib.metadata.version("datasets"),
        "pyyaml": importlib.metadata.version("PyYAML"),
        "cuda_visible_devices": "",
        "offline": True,
        "model_instantiated": False,
    }


def _attest_renderer_and_all_e0_configs(repo_root: Path) -> dict[str, Any]:
    """Re-attest exact renderer sources and all 36 prompt-relevant configs."""

    from script.experiment.fpct_e1_prepare_input_lock import (
        _load_a5_prompt_contract,
        _verify_all_e0_prompt_configs,
    )

    renderer = attest_e0_renderer_identity(repo_root)
    if renderer.get("renderer_source_identity_attested") is not True:
        raise ValueError("A5 renderer source identity is not attested")
    contract = _load_a5_prompt_contract(repo_root)
    configs = _verify_all_e0_prompt_configs(repo_root, contract)
    if (
        configs.get("evaluation_config_count") != 36
        or configs.get("enable_thinking") is not False
        or not isinstance(configs.get("records_sha256"), str)
    ):
        raise ValueError("A5 all-config prompt attestation is incomplete")
    return {"renderer": renderer, "configs": configs}


def _verify_streaming_evidence(value: Mapping[str, Any], repo_root: Path) -> None:
    from script.analysis.fpct_e1_streaming_synthetic_gate import (
        BUFFER_COPY_MULTIPLIER,
        PHYSICAL_CHUNK_ROWS,
        STREAMING_SCHEMA,
        _canonical_full_schema_row_bound,
    )

    streaming = value.get("streaming_stress", {})
    contract_checks = value.get("streaming_contract_checks", {})
    if any(contract_checks.get(name) is not True for name in STREAMING_CONTRACT_CHECK_NAMES):
        raise ValueError("A5 detailed streaming contract checks are incomplete")
    baseline = streaming.get("baseline", {})
    stress = streaming.get("stress", {})
    for label, record in (("baseline", baseline), ("stress", stress)):
        if (
            record.get("logical_rows") != record.get("emitted_rows")
            or record.get("semantic_stream_sha256")
            != record.get("replay_semantic_stream_sha256")
        ):
            raise ValueError(f"A5 {label} streaming replay evidence changed")
    if not isinstance(stress.get("logical_rows"), int) or stress["logical_rows"] < 1_000_000:
        raise ValueError("A5 streaming stress no longer covers one million rows")
    canonical_bound = _canonical_full_schema_row_bound(
        sha256_file(repo_root / STREAMING_SCHEMA.relative_to(REPO_ROOT))
    )
    observed_bound = max(
        int(baseline.get("max_observed_canonical_row_bytes", 0)),
        int(stress.get("max_observed_canonical_row_bytes", 0)),
    )
    if observed_bound > canonical_bound:
        raise ValueError("A5 observed streaming row exceeds canonical bound")
    allowance = BUFFER_COPY_MULTIPLIER * PHYSICAL_CHUNK_ROWS * canonical_bound
    expected_threshold = int(baseline.get("peak_rss_bytes", 0)) + allowance
    if streaming.get("threshold_bytes") != expected_threshold:
        raise ValueError("A5 streaming RSS threshold was not derived prospectively")
    if (
        streaming.get("bounded_peak_rss") is not True
        or int(stress.get("peak_rss_bytes", expected_threshold + 1))
        > expected_threshold
    ):
        raise ValueError("A5 streaming stress exceeds the frozen RSS threshold")
    estimated = 1 << (canonical_bound - 1).bit_length()
    if value.get("estimated_physical_bytes_per_row") != estimated:
        raise ValueError("A5 streaming physical row estimate changed")


def verify_a5_gate(
    gate_path: Path, *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    value = json.loads(gate_path.read_text(encoding="utf-8"))
    validate_a5_schema_artifact(value, repo_root=repo_root)
    historical = repo_root / HISTORICAL_A4_GATE_RELATIVE
    if sha256_file(historical) != HISTORICAL_A4_GATE_SHA256:
        raise ValueError("historical A4 gate bytes changed")
    if value.get("historical_a4_gate_sha256") != HISTORICAL_A4_GATE_SHA256:
        raise ValueError("A5 gate historical A4 binding changed")
    if value.get("historical_a4_gate_verified_against_current_tree") is not False:
        raise ValueError("A5 gate tried to reuse A4 current-tree verification")
    if value.get("execution_tree_sha256") != _execution_tree_sha256(repo_root):
        raise ValueError("A5 gate tracked source/test tree is stale")
    if value.get("tracked_files_sha256") != _tracked_source_sha256(repo_root):
        raise ValueError("A5 gate tracked source/test path map is stale")
    if value.get("evidence_sha256") != _evidence_sha256(value):
        raise ValueError("A5 gate evidence SHA does not recompute")
    _require_offline_cpu_environment()
    environment = value.get("environment")
    if (
        environment != _current_environment()
        or value.get("environment_identity_sha256")
        != hashlib.sha256(canonical_json_bytes(environment)).hexdigest()
        or value.get("model_instantiated") is not False
        or value.get("model_forward_run") is not False
    ):
        raise ValueError("A5 gate environment/model firewall identity changed")
    _verify_streaming_evidence(value, repo_root)
    _attest_renderer_and_all_e0_configs(repo_root)
    instrumentation = _verify_instrumentation(repo_root)
    tracked = value.get("tracked_files_sha256", {})
    if any(tracked.get(path) != digest for path, digest in instrumentation.items()):
        raise ValueError("A5 gate does not bind all three instrumentation artifacts")
    return dict(value)


def _run_tests(repo_root: Path) -> tuple[int, str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-cov",
        "-p",
        "no:cacheprovider",
        "--basetemp=/tmp/pytest-e1-a5-prompt-gate",
        *TEST_FILES,
    ]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr
    matches = list(_PASSED_PATTERN.finditer(output))
    if not matches:
        raise RuntimeError("A5 pytest output does not report a passed count")
    return int(matches[-1].group("count")), hashlib.sha256(output.encode()).hexdigest()


def _run_streaming_oracles() -> dict[str, Any]:
    # Re-execute production writer/replay under A5 sources.  The old A4 receipt
    # itself is intentionally not verified against this changed tree.
    from script.analysis.fpct_e1_streaming_synthetic_gate import (
        BUFFER_COPY_MULTIPLIER,
        PHYSICAL_CHUNK_ROWS,
        STREAMING_SCHEMA,
        _canonical_full_schema_row_bound,
        _fresh_worker,
    )

    baseline = _fresh_worker("a5_two_chunk", 10)
    canonical_bound = _canonical_full_schema_row_bound(sha256_file(STREAMING_SCHEMA))
    allowance = BUFFER_COPY_MULTIPLIER * PHYSICAL_CHUNK_ROWS * canonical_bound
    threshold = int(baseline["peak_rss_bytes"]) + allowance
    # The observation-derived baseline and schema-derived allowance are frozen
    # before the million-row worker starts.
    stress = _fresh_worker("a5_million_row", 2233)
    for record in (baseline, stress):
        if (
            record["logical_rows"] != record["emitted_rows"]
            or record["semantic_stream_sha256"]
            != record["replay_semantic_stream_sha256"]
        ):
            raise RuntimeError("A5 streaming semantic replay failed")
    if stress["logical_rows"] < 1_000_000:
        raise RuntimeError("A5 streaming stress did not cover one million rows")
    observed_bound = max(
        int(baseline["max_observed_canonical_row_bytes"]),
        int(stress["max_observed_canonical_row_bytes"]),
    )
    if observed_bound > canonical_bound:
        raise RuntimeError("A5 observed streaming row exceeds canonical bound")
    if int(stress["peak_rss_bytes"]) > threshold:
        raise RuntimeError("A5 streaming stress exceeds preregistered RSS threshold")
    estimated = 1 << (canonical_bound - 1).bit_length()
    case_fields = (
        "logical_rows",
        "emitted_rows",
        "semantic_stream_sha256",
        "replay_semantic_stream_sha256",
        "chunk_count",
        "physical_file_bytes",
        "max_physical_chunk_bytes",
        "max_observed_canonical_row_bytes",
        "peak_rss_bytes",
    )
    return {
        "baseline": {field: baseline[field] for field in case_fields},
        "stress": {field: stress[field] for field in case_fields},
        "threshold_bytes": threshold,
        "bounded_peak_rss": True,
        "estimated_physical_bytes_per_row": estimated,
    }


def build_gate(*, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    _require_offline_cpu_environment()
    historical = repo_root / HISTORICAL_A4_GATE_RELATIVE
    if sha256_file(historical) != HISTORICAL_A4_GATE_SHA256:
        raise ValueError("historical A4 gate bytes changed")
    _attest_renderer_and_all_e0_configs(repo_root)
    instrumentation = _verify_instrumentation(repo_root)
    streaming_evidence = _run_streaming_oracles()
    passed, pytest_output_sha256 = _run_tests(repo_root)
    environment = _current_environment()
    checks = {
        "a4_streaming_oracles_pass": True,
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
        "renderer_source_identity_attested": True,
    }
    value = {
        "schema_version": 7,
        "protocol_id": A5_PROTOCOL_ID,
        "artifact_type": "a5_prompt_synthetic_gate",
        "status": "GO_PRE_NATURAL_SYNTHETIC_HARD_GATE",
        "execution_tree_sha256": _execution_tree_sha256(repo_root),
        "tracked_files_sha256": _tracked_source_sha256(repo_root),
        "historical_a4_gate_sha256": HISTORICAL_A4_GATE_SHA256,
        "historical_a4_gate_verified_against_current_tree": False,
        "tests": {
            "passed": passed,
            "failed": 0,
            "output_sha256": pytest_output_sha256,
        },
        "streaming_stress": {
            key: value
            for key, value in streaming_evidence.items()
            if key != "estimated_physical_bytes_per_row"
        },
        "streaming_contract_checks": {
            name: True for name in STREAMING_CONTRACT_CHECK_NAMES
        },
        "estimated_physical_bytes_per_row": streaming_evidence[
            "estimated_physical_bytes_per_row"
        ],
        "environment": environment,
        "environment_identity_sha256": hashlib.sha256(
            canonical_json_bytes(environment)
        ).hexdigest(),
        "checks": checks,
        "natural_e0_design_accessed": False,
        "model_instantiated": False,
        "model_or_checkpoint_loaded": False,
        "model_forward_run": False,
        "gpu_or_kubernetes_used": False,
        "e1_pilot_consumed": False,
    }
    value["evidence_sha256"] = _evidence_sha256(value)
    validate_a5_schema_artifact(value, repo_root=repo_root)
    if any(
        value["tracked_files_sha256"].get(path) != digest
        for path, digest in instrumentation.items()
    ):
        raise AssertionError("A5 instrumentation artifacts escaped tracked closure")
    _verify_streaming_evidence(value, repo_root)
    return value


def publish_gate(
    value: Mapping[str, Any], output: Path, *, repo_root: Path = REPO_ROOT
) -> None:
    validate_a5_schema_artifact(value, repo_root=repo_root)
    payload = canonical_json_bytes(value) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable A5 gate")
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / A5_GATE_RELATIVE)
    args = parser.parse_args(argv)
    _require_offline_cpu_environment()
    value = build_gate(repo_root=REPO_ROOT)
    publish_gate(value, args.output, repo_root=REPO_ROOT)
    verify_a5_gate(args.output, repo_root=REPO_ROOT)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
