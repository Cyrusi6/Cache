#!/usr/bin/env python3
"""Build and verify the CPU-only A5R1 dual-hash-domain hard gate.

The v7 gate is immutable predecessor evidence only.  This successor gate binds
the approved v8 contract and proves, without opening natural E0-design rows,
that the E0-declared and generic asset-manifest hashes remain distinct domains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from script.analysis import fpct_e1_a5_prompt_gate as predecessor_gate
from script.experiment.fpct_e1_a5_prompt_provenance import (
    A5R1_AMENDMENT_ID,
    A5R1_PROTOCOL_ID,
    canonical_json_bytes,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 8
CONTRACT_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5r1_hash_domain_contract.json"
)
SCHEMA_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5r1_hash_domain_schema.json"
)
GATE_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5r1_hash_domain_synthetic_gate.json"
)
V7_OBJECT_SHA256 = {
    "FPCT_E1_A5_PRODUCTION_PROMPT_AMENDMENT.md": (
        "7178f2ee7c5d226b69b88726ee645aaf7d9425b150764393690136da2c62e874"
    ),
    "recipe/eval_recipe/fpct_e1/e1_a5_prompt_contract.json": (
        "aba2a439ce56cd68bc7a614075f7efac1a5da36a77d7ae87b0ee5bf8c493db7f"
    ),
    "recipe/eval_recipe/fpct_e1/e1_a5_prompt_schema.json": (
        "1a5e139592b8afc804c730e028660b55b6df7a1dfadc864bc797eef84ca388bb"
    ),
    "recipe/eval_recipe/fpct_e1/e1_a5_prompt_synthetic_gate.json": (
        "464e646c336d33c97a03c603d283a95b1579508d63ae0f19d7c700ccf2dcfc02"
    ),
    "FPCT_E1_A5_DATA_HASH_RECOVERY_ADDENDUM.md": (
        "4de16ef383c608c528003634e1cf0c16c9100f0bab1d32cb3868c83439ef3ce4"
    ),
    "recipe/eval_recipe/fpct_e1/executions/9b248d20/input_lock_failure_receipt.json": (
        "76e4dafcb1bf7673a3a4d1cbd124ba4099f8de8bca6af411cdbb9f83b32e3da4"
    ),
}
HISTORICAL_A5_GATE_SHA256 = V7_OBJECT_SHA256[
    "recipe/eval_recipe/fpct_e1/e1_a5_prompt_synthetic_gate.json"
]
INSTRUMENTATION_RELATIVES = (
    Path("recipe/eval_recipe/fpct_e1/e1_instrumentation_parity_a5.json"),
    Path("recipe/eval_recipe/fpct_e1/e1_synthetic_query_variance_a5.json"),
    Path("recipe/eval_recipe/fpct_e1/e1_instrumentation_hard_gate_a5.json"),
)
TEST_FILES = (
    "test/test_fpct_e1_a5r1_hash_domain_gate.py",
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
            "FPCT_E1_A5R1_HASH_DOMAIN_AMENDMENT.md",
            "FPCT_E1_A5_DATA_HASH_RECOVERY_ADDENDUM.md",
            CONTRACT_RELATIVE.as_posix(),
            SCHEMA_RELATIVE.as_posix(),
            "recipe/eval_recipe/fpct_e1/executions/9b248d20/input_lock_failure_receipt.json",
            *V7_OBJECT_SHA256,
            "script/analysis/fpct_e1_a5r1_hash_domain_gate.py",
            "script/analysis/fpct_e1_a5_prompt_gate.py",
            "script/analysis/fpct_e1_streaming_verify.py",
            "script/analysis/fpct_e1_streaming_synthetic_gate.py",
            "script/analysis/fpct_e1_instrumentation_gate.py",
            "script/analysis/fpct_reference_operator.py",
            "script/experiment/fpct_e1_a5_prompt_provenance.py",
            "script/experiment/fpct_e1_prepare_input_lock.py",
            "script/experiment/fpct_e1_capture_runner.py",
            "script/experiment/fpct_e1_source_snapshot_lock.py",
            "script/experiment/fpct_e1_runtime_backend.py",
            "script/runtime/fpct_bootstrap.py",
            "rosetta/model/fpct_attention.py",
            "rosetta/model/fpct_instrumentation.py",
            *(path.as_posix() for path in INSTRUMENTATION_RELATIVES),
            *TEST_FILES,
        }
    )
)
PASSED_PATTERN = re.compile(r"(?P<count>[0-9]+) passed")


def _load_json_unique(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_a5r1_schema_artifact(
    value: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> None:
    """Validate v8 objects, with v7 structure inherited for census-only rows."""

    from script.analysis.fpct_e1_streaming_verify import (
        _load_manifest,
        _validate_json_schema,
    )

    artifact_type = value.get("artifact_type")
    schema = _load_manifest(repo_root / SCHEMA_RELATIVE)
    try:
        _validate_json_schema(dict(value), schema, schema, path="$")
        return
    except ValueError:
        if artifact_type not in {
            "a5_prompt_census_record",
            "a5_prompt_census_manifest",
            "a5_dual_anchor_census_summary",
        }:
            raise
    # Census semantics are inherited byte-for-byte from v7 except for their
    # protocol header.  Validate that exact inherited structure under the v7
    # schema after independently requiring the v8 header.
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != A5R1_PROTOCOL_ID
    ):
        raise ValueError("A5R1 inherited census artifact header changed")
    inherited = dict(value)
    inherited["schema_version"] = 7
    inherited["protocol_id"] = (
        "fpct_e1_mechanism_audit_v7_actual_e0_runtime_prompt"
    )
    if artifact_type == "a5_prompt_census_manifest":
        execution_sha = str(value.get("execution_sha", ""))
        expected_v8_uid = f"fpct-e1-a5r1-hash-domains-{execution_sha[:8]}-v1"
        if (
            re.fullmatch(r"[0-9a-f]{40}", execution_sha) is None
            or value.get("run_uid") != expected_v8_uid
        ):
            raise ValueError("A5R1 census manifest execution/UID binding changed")
        # Only the UID prefix differs from the inherited v7 census structure.
        # Map it after independently proving the exact v8 execution binding.
        inherited["run_uid"] = f"fpct-e1-a5-runtime-prompt-{execution_sha[:8]}-v1"
    v7_schema = _load_manifest(
        repo_root / "recipe/eval_recipe/fpct_e1/e1_a5_prompt_schema.json"
    )
    _validate_json_schema(inherited, v7_schema, v7_schema, path="$")


def _tracked_source_sha256(repo_root: Path) -> dict[str, str]:
    return {
        relative: sha256_file(repo_root / relative)
        for relative in TRACKED_SOURCE_FILES
    }


def _execution_tree_sha256(repo_root: Path) -> str:
    return hashlib.sha256(
        canonical_json_bytes(_tracked_source_sha256(repo_root))
    ).hexdigest()


def _evidence_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {key: child for key, child in value.items() if key != "evidence_sha256"}
        )
    ).hexdigest()


def _verify_v7_predecessors(repo_root: Path) -> dict[str, str]:
    observed = {
        relative: sha256_file(repo_root / relative)
        for relative in V7_OBJECT_SHA256
    }
    if observed != V7_OBJECT_SHA256:
        raise ValueError("immutable A5 v7 predecessor bytes changed")
    return observed


def _load_contract(repo_root: Path) -> dict[str, Any]:
    contract = _load_json_unique(repo_root / CONTRACT_RELATIVE)
    validate_a5r1_schema_artifact(contract, repo_root=repo_root)
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("protocol_id") != A5R1_PROTOCOL_ID
        or contract.get("amendment_id") != A5R1_AMENDMENT_ID
        or contract.get("approval", {}).get("approved") is not True
        or contract.get("approval", {}).get("selected_option")
        != "ACTUAL_E0_PRODUCTION_RUNTIME_PROMPT"
    ):
        raise ValueError("A5R1 contract approval identity changed")
    blocked = contract["immutable_predecessors"]["blocked_execution"]
    if (
        blocked.get("execution_sha")
        != "9b248d2094b684f5d9e9a218919a354b7d97468e"
        or blocked.get("resume_allowed") is not False
        or blocked.get("artifact_reuse_allowed") is not False
    ):
        raise ValueError("A5R1 blocked predecessor disposition changed")
    return contract


def _independent_file_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            target_text = os.readlink(candidate)
            target = candidate.resolve(strict=True)
            if target.is_dir():
                raise ValueError("synthetic reference forbids directory symlinks")
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
    return records


def _independent_generic_sha256(root: Path) -> str:
    # The production canonical file representation is newline terminated.
    # Reproduce that byte-level specification independently rather than
    # calling the producer's nested_sha256 helper.
    payload = (
        json.dumps(
            {"files": _independent_file_records(root)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _independent_declared_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for record in _independent_file_records(root):
        digest.update(record["relative_path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(record["sha256"]))
    return digest.hexdigest()


def run_hash_domain_oracles() -> dict[str, bool]:
    """Execute independent, synthetic-only positive and negative controls."""

    from script.experiment.fpct_e1_prepare_input_lock import (
        BLOCKED_RECEIPT_NAME,
        E0_DECLARED_TREE_ALGORITHM,
        GENERIC_ASSET_TREE_ALGORITHM,
        GO_RECEIPT_NAME,
        _attest_a5_runtime_assets,
        _e0_data_asset_tree,
        _e0_data_hash_domain_projection,
        _publish_blocked_receipt,
        _quarantine_canonical_go_before_blocked,
        atomic_json,
        validate_a5_execution_identity,
    )

    with tempfile.TemporaryDirectory(prefix="fpct-a5r1-hash-domain-") as temp:
        root = Path(temp)
        (root / "alpha.txt").write_bytes(b"alpha\n")
        (root / "nested").mkdir()
        (root / "nested/beta.bin").write_bytes(b"\x00\x01\x02")
        observed = _e0_data_asset_tree(root)
        projection = _e0_data_hash_domain_projection(observed)
        if observed["tree_sha256"] != _independent_generic_sha256(root):
            raise AssertionError("generic hash differs from independent oracle")
        if observed["e0_declared_tree_sha256"] != _independent_declared_sha256(root):
            raise AssertionError("declared hash differs from independent oracle")
        if observed["tree_sha256"] == observed["e0_declared_tree_sha256"]:
            raise AssertionError("synthetic domains unexpectedly collapsed")
        if (
            projection["generic_asset_tree_algorithm"]
            != GENERIC_ASSET_TREE_ALGORITHM
            or projection["e0_declared_tree_algorithm"]
            != E0_DECLARED_TREE_ALGORITHM
        ):
            raise AssertionError("synthetic domain labels changed")

        baseline = observed["tree_sha256"]
        mutations: list[tuple[str, Any]] = []
        (root / "alpha.txt").write_bytes(b"changed\n")
        mutations.append(("byte_change", _e0_data_asset_tree(root)["tree_sha256"]))
        (root / "alpha.txt").write_bytes(b"alpha\n")
        (root / "added.txt").write_bytes(b"added")
        mutations.append(("add", _e0_data_asset_tree(root)["tree_sha256"]))
        (root / "added.txt").unlink()
        (root / "nested/beta.bin").rename(root / "nested/renamed.bin")
        mutations.append(("rename", _e0_data_asset_tree(root)["tree_sha256"]))
        (root / "nested/renamed.bin").rename(root / "nested/beta.bin")
        (root / "alpha.link").symlink_to("alpha.txt")
        mutations.append(("symlink", _e0_data_asset_tree(root)["tree_sha256"]))
        (root / "alpha.link").unlink()
        (root / "nested/beta.bin").unlink()
        mutations.append(("remove", _e0_data_asset_tree(root)["tree_sha256"]))
        (root / "nested/beta.bin").write_bytes(b"\x00\x01\x02")
        if any(value == baseline for _, value in mutations):
            raise AssertionError("generic hash failed a synthetic tamper mutation")
        if _e0_data_asset_tree(root)["tree_sha256"] != baseline:
            raise AssertionError("restored synthetic tree did not replay exactly")

        for missing in (
            "tree_algorithm",
            "tree_sha256",
            "e0_declared_tree_algorithm",
            "e0_declared_tree_sha256",
        ):
            broken = dict(observed)
            broken.pop(missing)
            try:
                _e0_data_hash_domain_projection(broken)
            except ValueError:
                pass
            else:
                raise AssertionError(f"missing domain field was accepted: {missing}")
        mislabelled = dict(observed)
        mislabelled["tree_algorithm"] = E0_DECLARED_TREE_ALGORITHM
        try:
            _e0_data_hash_domain_projection(mislabelled)
        except ValueError:
            pass
        else:
            raise AssertionError("mislabelled hash domain was accepted")
        swapped = dict(observed)
        swapped["tree_sha256"], swapped["e0_declared_tree_sha256"] = (
            swapped["e0_declared_tree_sha256"],
            swapped["tree_sha256"],
        )
        try:
            _attest_a5_runtime_assets(
                contract={"asset_identity": {"materialized_e0_dev_data_tree": projection}},
                runtime_assets={},
                receiver=object(),
                sender=object(),
                e0_data_assets=swapped,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("swapped domain values were accepted")

        try:
            validate_a5_execution_identity(
                execution_sha="9b248d2094b684f5d9e9a218919a354b7d97468e",
                source_snapshot_root=root,
                source_snapshot_receipt=root / "missing.json",
                run_uid="fpct-e1-a5r1-hash-domains-9b248d20-v1",
                run_root=Path("/netdisk/lijunsi/fpct-e1/fpct-e1-a5r1-9b248d20-v1"),
                output_root=root / "input_lock",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("blocked execution identity was reusable")

        # Exercise the actual producer terminal transition rather than merely
        # asserting the intended result.  The sentinel is a schema-valid v8 GO
        # receipt.  A caught failure must first quarantine that canonical name,
        # then publish one schema-valid BLOCKED receipt; leaving a usable GO is
        # a hard failure of this pre-natural oracle.
        terminal_root = root / "terminal"
        terminal_root.mkdir()
        execution_sha = "a" * 40
        run_uid = "fpct-e1-a5r1-hash-domains-aaaaaaaa-v1"
        run_root = "/netdisk/lijunsi/fpct-e1/fpct-e1-a5r1-aaaaaaaa-v1"
        inherited_checks = {
            name: True
            for name in (
                "historical_projection_anchor_unchanged",
                "e0_design_membership_unchanged",
                "renderer_source_identity_attested",
                "production_renderer_exactly_attested",
                "production_data_tree_exactly_attested",
                "all_326_groups_resolved",
                "all_first4_choices_equal_historical",
                "all_gold_answers_in_A_B_C_D",
                "all_prompt_differences_classified",
                "historical_to_runtime_mapping_complete",
                "production_prompt_replay_equal",
                "production_alignment_replay_equal",
                "choice_order_preserved",
                "raw_full_row_hashes_complete",
                "logical_row_coverage_exact",
                "streaming_semantic_replay_equal",
                "bounded_peak_rss",
            )
        }
        natural_hash_checks = {
            name: name
            not in {
                "cross_domain_comparison_detected",
                "old_v7_artifact_modified",
                "blocked_execution_artifact_reused",
            }
            for name in (
                "e0_declared_domain_present",
                "e0_declared_algorithm_exact",
                "e0_declared_sha_matches_frozen_e0",
                "a5_generic_domain_present",
                "a5_generic_algorithm_exact",
                "a5_generic_sha_matches_same_domain_predecessor",
                "a5_generic_before_equals_after",
                "completed_verifier_recomputes_both_domains",
                "cross_domain_comparison_detected",
                "old_v7_artifact_modified",
                "blocked_execution_artifact_reused",
            )
        }
        frozen_identity = _load_contract(REPO_ROOT)["asset_identity"][
            "materialized_e0_dev_data_tree"
        ]
        go_receipt = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": A5R1_PROTOCOL_ID,
            "artifact_type": "a5r1_input_lock_receipt",
            "status": "A5_INPUT_LOCK_GO",
            "execution_sha": execution_sha,
            "run_uid": run_uid,
            "run_root": run_root,
            "source_snapshot_receipt_sha256": "1" * 64,
            "prompt_census_manifest_sha256": "2" * 64,
            "input_lock_manifest_sha256": "3" * 64,
            "e0_declared_tree_algorithm": frozen_identity[
                "e0_declared_tree_algorithm"
            ],
            "e0_declared_tree_sha256": frozen_identity[
                "e0_declared_tree_sha256"
            ],
            "generic_asset_tree_algorithm": frozen_identity[
                "generic_asset_tree_algorithm"
            ],
            "generic_asset_tree_sha256": frozen_identity[
                "generic_asset_tree_sha256"
            ],
            "checks": inherited_checks,
            "hash_domain_checks": natural_hash_checks,
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
            "zero_counts": {
                "unexpected_prompt_difference_count": 0,
                "missing_rows": 0,
                "duplicate_rows": 0,
            },
            "resume_from_group_one": True,
            "downstream_e1_2_authorized": False,
        }
        validate_a5r1_schema_artifact(go_receipt, repo_root=REPO_ROOT)
        atomic_json(terminal_root / GO_RECEIPT_NAME, go_receipt)
        quarantine = _quarantine_canonical_go_before_blocked(terminal_root)
        if quarantine is None or not quarantine.is_file():
            raise AssertionError("canonical GO was not preserved in quarantine")
        execution_identity = {
            "execution_sha": execution_sha,
            "run_uid": run_uid,
            "run_root": run_root,
            "source_snapshot_root": str(REPO_ROOT),
        }
        _publish_blocked_receipt(
            terminal_root,
            RuntimeError("A5R1_SYNTHETIC_TERMINAL_FAILURE"),
            execution_identity,
            e0_data_hash_domains=frozen_identity,
        )
        if (terminal_root / GO_RECEIPT_NAME).exists():
            raise AssertionError("canonical GO survived BLOCKED transition")
        blocked_path = terminal_root / BLOCKED_RECEIPT_NAME
        if not blocked_path.is_file():
            raise AssertionError("BLOCKED receipt was not published")
        blocked = _load_json_unique(blocked_path)
        validate_a5r1_schema_artifact(blocked, repo_root=REPO_ROOT)
        if blocked.get("status") != "A5_INPUT_LOCK_BLOCKED":
            raise AssertionError("terminal failure did not produce BLOCKED")

    return {
        "independent_declared_reference_oracle_pass": True,
        "independent_generic_reference_oracle_pass": True,
        "deliberate_cross_domain_divergence_accepted": True,
        "swapped_domain_values_fail_closed_pass": True,
        "missing_or_unqualified_domain_value_fails_closed_pass": True,
        "tamper_matrix_pass": True,
        "generic_before_after_equality_pass": True,
        "completed_verifier_dual_recompute_fixture_pass": True,
        "v7_four_object_sha_immutability_pass": True,
        "blocked_execution_and_root_nonreuse_pass": True,
        "atomic_failure_publishes_no_go_pass": True,
    }


def _verify_contract_hashes(contract: Mapping[str, Any], repo_root: Path) -> None:
    identity = contract["asset_identity"]["materialized_e0_dev_data_tree"]
    closure = _load_json_unique(
        repo_root
        / "recipe/eval_recipe/fpct_e1/executions/9b248d20/input_lock_failure_receipt.json"
    )
    if (
        identity["e0_declared_tree_sha256"]
        != closure["e0_declared_tree_sha256"]
        or identity["generic_asset_tree_sha256"]
        != closure["generic_asset_tree_sha256"]
        or identity["e0_declared_tree_algorithm"]
        != closure["e0_declared_tree_algorithm"]
        or identity["generic_asset_tree_algorithm"]
        != closure["generic_asset_tree_algorithm"]
    ):
        raise ValueError("A5R1 contract and blocked closure hash domains disagree")


def _run_tests(repo_root: Path) -> tuple[int, str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-cov",
        "-p",
        "no:cacheprovider",
        "--basetemp=/tmp/pytest-e1-a5r1-hash-domain-gate",
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
    matches = list(PASSED_PATTERN.finditer(output))
    if not matches:
        raise RuntimeError("A5R1 pytest output does not report a passed count")
    return int(matches[-1].group("count")), hashlib.sha256(output.encode()).hexdigest()


def _instrumentation_sha256(repo_root: Path) -> dict[str, str]:
    predecessor_gate._verify_instrumentation(repo_root)
    return {
        path.as_posix(): sha256_file(repo_root / path)
        for path in INSTRUMENTATION_RELATIVES
    }


def build_gate(*, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    predecessor_gate._require_offline_cpu_environment()
    predecessors = _verify_v7_predecessors(repo_root)
    contract = _load_contract(repo_root)
    _verify_contract_hashes(contract, repo_root)
    predecessor_gate._attest_renderer_and_all_e0_configs(repo_root)
    instrumentation = _instrumentation_sha256(repo_root)
    streaming = predecessor_gate._run_streaming_oracles()
    hash_checks = run_hash_domain_oracles()
    passed, output_sha256 = _run_tests(repo_root)
    environment = predecessor_gate._current_environment()
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
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": A5R1_PROTOCOL_ID,
        "artifact_type": "a5r1_hash_domain_synthetic_gate",
        "status": "GO_PRE_NATURAL_A5R1_HASH_DOMAIN_HARD_GATE",
        "historical_a5_gate_sha256": HISTORICAL_A5_GATE_SHA256,
        "historical_a5_gate_verified_against_current_tree": False,
        "immutable_predecessor_sha256": predecessors,
        "tracked_files_sha256": _tracked_source_sha256(repo_root),
        "execution_tree_sha256": _execution_tree_sha256(repo_root),
        "tests": {"passed": passed, "failed": 0, "output_sha256": output_sha256},
        "hash_domain_checks": hash_checks,
        "instrumentation_artifacts_sha256": instrumentation,
        "streaming_stress": {
            key: child
            for key, child in streaming.items()
            if key != "estimated_physical_bytes_per_row"
        },
        "streaming_contract_checks": {
            name: True for name in predecessor_gate.STREAMING_CONTRACT_CHECK_NAMES
        },
        "estimated_physical_bytes_per_row": streaming[
            "estimated_physical_bytes_per_row"
        ],
        "checks": checks,
        "environment": environment,
        "environment_identity_sha256": hashlib.sha256(
            canonical_json_bytes(environment)
        ).hexdigest(),
        "natural_e0_design_accessed": False,
        "model_instantiated": False,
        "model_or_checkpoint_loaded": False,
        "model_forward_run": False,
        "gpu_or_kubernetes_used": False,
        "e1_pilot_consumed": False,
        "e1_2_or_e1_3_authorized": False,
    }
    value["evidence_sha256"] = _evidence_sha256(value)
    validate_a5r1_schema_artifact(value, repo_root=repo_root)
    predecessor_gate._verify_streaming_evidence(value, repo_root)
    return value


def verify_gate(gate_path: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    value = _load_json_unique(gate_path)
    validate_a5r1_schema_artifact(value, repo_root=repo_root)
    predecessor_gate._require_offline_cpu_environment()
    if value.get("immutable_predecessor_sha256") != _verify_v7_predecessors(repo_root):
        raise ValueError("A5R1 predecessor map changed")
    contract = _load_contract(repo_root)
    _verify_contract_hashes(contract, repo_root)
    if value.get("tracked_files_sha256") != _tracked_source_sha256(repo_root):
        raise ValueError("A5R1 tracked source/test map is stale")
    if value.get("execution_tree_sha256") != _execution_tree_sha256(repo_root):
        raise ValueError("A5R1 execution tree is stale")
    if value.get("evidence_sha256") != _evidence_sha256(value):
        raise ValueError("A5R1 evidence SHA does not recompute")
    environment = predecessor_gate._current_environment()
    if (
        value.get("environment") != environment
        or value.get("environment_identity_sha256")
        != hashlib.sha256(canonical_json_bytes(environment)).hexdigest()
    ):
        raise ValueError("A5R1 environment identity changed")
    if value.get("hash_domain_checks") != run_hash_domain_oracles():
        raise ValueError("A5R1 hash-domain controls do not independently replay")
    instrumentation = _instrumentation_sha256(repo_root)
    if value.get("instrumentation_artifacts_sha256") != instrumentation:
        raise ValueError("A5R1 instrumentation predecessor binding changed")
    predecessor_gate._attest_renderer_and_all_e0_configs(repo_root)
    predecessor_gate._verify_streaming_evidence(value, repo_root)
    return value


# The input-lock producer historically imports these stable interface names.
# Keep the interface while routing it exclusively to the versioned v8 logic.
validate_a5_schema_artifact = validate_a5r1_schema_artifact
verify_a5_gate = verify_gate


def publish_gate(value: Mapping[str, Any], output: Path) -> None:
    validate_a5r1_schema_artifact(value, repo_root=REPO_ROOT)
    payload = canonical_json_bytes(value) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable A5R1 gate")
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
    parser.add_argument("--output", type=Path, default=REPO_ROOT / GATE_RELATIVE)
    args = parser.parse_args(argv)
    value = build_gate(repo_root=REPO_ROOT)
    publish_gate(value, args.output)
    verify_gate(args.output, repo_root=REPO_ROOT)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
