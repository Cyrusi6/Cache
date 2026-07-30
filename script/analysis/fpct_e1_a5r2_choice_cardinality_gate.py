#!/usr/bin/env python3
"""Build the CPU-only, pre-natural A5R2 choice-cardinality hard gate.

This gate is deliberately unable to open an E0-design row.  It binds the
prospective v9 contract, re-attests the immutable v8 predecessor, executes
synthetic choice/cardinality/taxonomy oracles and publishes one transactional
evidence object.  Natural audit and input-lock code may consume the gate only
after the containing Git commit is clean and pushed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from script.experiment.fpct_e1_a5_prompt_provenance import (
    A5R2_AMENDMENT_ID,
    A5R2_PROTOCOL_ID,
    build_choice_cardinality_audit_lock_v9,
    canonical_historical_content_sha256,
    canonical_json_bytes,
    choice_audit_projection_v9,
    choice_cardinality_record_v9,
    choice_payload_v9,
    historical_projected_example_v9,
    sha256_file,
    summarize_choice_cardinality_audit_v9,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 9
BASE_COMMIT = "f782128774dd88081194c2873efd707044d1db2c"
BLOCKED_EXECUTION_SHA = "37be816ad611b8b0d916bd98c840c5f31efe2b50"
BLOCKED_CLOSURE_SHA256 = (
    "ec3ae949b557da957a1a1f295f41b91b8522420445b5479a945b1f59b5de8e8a"
)
AMENDMENT_RELATIVE = Path("FPCT_E1_A5R2_CHOICE_CARDINALITY_AMENDMENT.md")
CONTRACT_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5r2_choice_cardinality_contract.json"
)
SCHEMA_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5r2_choice_cardinality_schema.json"
)
GATE_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5r2_choice_cardinality_synthetic_gate.json"
)
BLOCKED_CLOSURE_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/executions/37be816a/input_lock_failure_receipt.json"
)
V8_OBJECT_SHA256 = {
    "FPCT_E1_A5R1_HASH_DOMAIN_AMENDMENT.md": (
        "8eafb29d3d736740730e10505ebd8217e779e5ef4d3a128cfb0db3a10a018068"
    ),
    "recipe/eval_recipe/fpct_e1/e1_a5r1_hash_domain_contract.json": (
        "643151b67d98c84c1120b52705b0fe837fb4106664f10200ada1a692c744a0b1"
    ),
    "recipe/eval_recipe/fpct_e1/e1_a5r1_hash_domain_schema.json": (
        "7bd2478f7ef3cfd32e752056cf161b8575b84a1f65088c84a0d2c37aec43704b"
    ),
    "recipe/eval_recipe/fpct_e1/e1_a5r1_hash_domain_synthetic_gate.json": (
        "a2e53784fa46d2f63963bdb41e327a4e17936cce2bff765802b34f3d87869a9f"
    ),
}
POPULATION_SOURCE_SHA256 = {
    "recipe/eval_recipe/fpct_e1/e1_data_split_manifest.json": (
        "030b4236ed9bec82b145227259733b32a8c76af63adf2fa0f1282e3638b5b11d"
    ),
    "recipe/eval_recipe/fpct_e0/exploratory_dev_manifest.json": (
        "25fe8c4dceeaa1174e1433a02ec86f312d909d7d58c3c9a8c7f2caa9d908216a"
    ),
}
TEST_FILES = (
    "test/test_fpct_e1_a5r2_choice_cardinality_gate.py",
    "test/test_fpct_e1_a5r2_choice_cardinality.py",
    "test/test_fpct_e1_a5r2_prepare_integration.py",
    "test/test_fpct_e1_prepare_input_lock.py",
    "test/test_fpct_e1_a5_prompt_provenance.py",
    "test/test_fpct_e1_a5_instrumentation_static.py",
    "test/test_fpct_e1_instrumentation_gate_a5.py",
    "test/test_fpct_e1_streaming.py",
    "test/test_fpct_e1_streaming_synthetic_gate.py",
    "test/test_fpct_sealed_import.py",
    "test/test_fpct_e1_source_snapshot_lock.py",
)
TRACKED_SOURCE_FILES = tuple(
    sorted(
        {
            AMENDMENT_RELATIVE.as_posix(),
            CONTRACT_RELATIVE.as_posix(),
            SCHEMA_RELATIVE.as_posix(),
            BLOCKED_CLOSURE_RELATIVE.as_posix(),
            *V8_OBJECT_SHA256,
            *POPULATION_SOURCE_SHA256,
            "script/analysis/fpct_e1_a5r2_choice_cardinality_gate.py",
            "script/analysis/fpct_e1_a5r1_hash_domain_gate.py",
            "script/analysis/fpct_e1_a5_prompt_gate.py",
            "script/analysis/fpct_e1_streaming_verify.py",
            "script/analysis/fpct_e1_streaming_synthetic_gate.py",
            "script/analysis/fpct_e1_instrumentation_gate.py",
            "script/experiment/fpct_e1_a5_prompt_provenance.py",
            "script/experiment/fpct_e1_prepare_input_lock.py",
            "script/experiment/fpct_e1_capture_runner.py",
            "script/experiment/fpct_e1_source_snapshot_lock.py",
            "script/experiment/fpct_e1_runtime_backend.py",
            "script/runtime/fpct_bootstrap.py",
            *(str(path) for path in TEST_FILES),
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


def validate_a5r2_schema_artifact(
    value: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> None:
    from script.analysis.fpct_e1_streaming_verify import (
        _load_manifest,
        _validate_json_schema,
    )

    schema = _load_manifest(repo_root / SCHEMA_RELATIVE)
    _validate_json_schema(dict(value), schema, schema, path="$")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_binding(repo_root: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": sha256_file(repo_root / relative)}


def _tracked_file_bindings(repo_root: Path) -> list[dict[str, str]]:
    return [_file_binding(repo_root, relative) for relative in TRACKED_SOURCE_FILES]


def _execution_tree_sha256(repo_root: Path) -> str:
    return _sha256_bytes(canonical_json_bytes(_tracked_file_bindings(repo_root)))


def _evidence_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        canonical_json_bytes(
            {key: child for key, child in value.items() if key != "evidence_sha256"}
        )
    )


def _require_offline_cpu_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("A5R2 gate requires CUDA_VISIBLE_DEVICES='' ")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"A5R2 gate requires {name}=1")


def _verify_predecessors(repo_root: Path) -> dict[str, str]:
    observed = {
        relative: sha256_file(repo_root / relative) for relative in V8_OBJECT_SHA256
    }
    if observed != V8_OBJECT_SHA256:
        raise ValueError("immutable A5R1 v8 predecessor bytes changed")
    if sha256_file(repo_root / BLOCKED_CLOSURE_RELATIVE) != BLOCKED_CLOSURE_SHA256:
        raise ValueError("immutable 37be816a failure closure changed")
    closure = _load_json_unique(repo_root / BLOCKED_CLOSURE_RELATIVE)
    if (
        closure.get("execution_sha") != BLOCKED_EXECUTION_SHA
        or closure.get("resume_allowed") is not False
        or closure.get("reuse_allowed") is not False
    ):
        raise ValueError("37be816a disposition changed")
    return observed


def _inherited_v8_streaming_evidence(repo_root: Path) -> dict[str, Any]:
    """Project only the immutable v8 resource evidence into the v9 gate."""

    relative = "recipe/eval_recipe/fpct_e1/e1_a5r1_hash_domain_synthetic_gate.json"
    path = repo_root / relative
    if sha256_file(path) != V8_OBJECT_SHA256[relative]:
        raise ValueError("immutable v8 streaming-evidence source changed")
    predecessor = _load_json_unique(path)
    evidence = {
        "source_gate_path": relative,
        "source_gate_sha256": V8_OBJECT_SHA256[relative],
        "evidence_role": (
            "IMMUTABLE_V8_PREDECESSOR_RESOURCE_EVIDENCE_NOT_V9_REMEASUREMENT"
        ),
        "estimated_physical_bytes_per_row": predecessor[
            "estimated_physical_bytes_per_row"
        ],
        "streaming_stress": predecessor["streaming_stress"],
        "streaming_contract_checks": predecessor["streaming_contract_checks"],
        "cross_resource_or_hash_domain_substitution_allowed": False,
    }
    if (
        evidence["estimated_physical_bytes_per_row"] != 4096
        or evidence["streaming_stress"].get("bounded_peak_rss") is not True
        or not all(evidence["streaming_contract_checks"].values())
    ):
        raise ValueError("immutable v8 streaming evidence is incomplete")
    return evidence


def _run_inherited_hash_domain_oracles() -> None:
    """Replay v8 dual-domain semantics without invoking the historical receipt API."""

    from script.analysis.fpct_e1_a5r1_hash_domain_gate import (
        _independent_declared_sha256,
        _independent_generic_sha256,
    )
    from script.experiment.fpct_e1_prepare_input_lock import (
        E0_DECLARED_TREE_ALGORITHM,
        GENERIC_ASSET_TREE_ALGORITHM,
        _attest_a5_runtime_assets,
        _e0_data_asset_tree,
        _e0_data_hash_domain_projection,
    )

    with tempfile.TemporaryDirectory(prefix="fpct-a5r2-hash-domains-") as temporary:
        root = Path(temporary)
        (root / "alpha.txt").write_bytes(b"alpha\n")
        (root / "nested").mkdir()
        (root / "nested/beta.bin").write_bytes(b"\x00\x01\x02")
        observed = _e0_data_asset_tree(root)
        projection = _e0_data_hash_domain_projection(observed)
        if (
            observed["tree_sha256"] != _independent_generic_sha256(root)
            or observed["e0_declared_tree_sha256"]
            != _independent_declared_sha256(root)
            or observed["tree_sha256"] == observed["e0_declared_tree_sha256"]
            or projection["generic_asset_tree_algorithm"]
            != GENERIC_ASSET_TREE_ALGORITHM
            or projection["e0_declared_tree_algorithm"]
            != E0_DECLARED_TREE_ALGORITHM
        ):
            raise AssertionError("A5R2 inherited dual hash-domain oracle changed")
        baseline = observed["tree_sha256"]
        (root / "alpha.txt").write_bytes(b"changed\n")
        if _e0_data_asset_tree(root)["tree_sha256"] == baseline:
            raise AssertionError("generic asset-domain tamper was not detected")
        (root / "alpha.txt").write_bytes(b"alpha\n")
        if _e0_data_asset_tree(root)["tree_sha256"] != baseline:
            raise AssertionError("restored generic asset domain did not replay")
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
                raise AssertionError(f"missing dual-domain field was accepted: {missing}")
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
            raise AssertionError("cross-domain swap was accepted")


def _load_contract(repo_root: Path) -> dict[str, Any]:
    contract = _load_json_unique(repo_root / CONTRACT_RELATIVE)
    validate_a5r2_schema_artifact(contract, repo_root=repo_root)
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("protocol_id") != A5R2_PROTOCOL_ID
        or contract.get("amendment_id") != A5R2_AMENDMENT_ID
        or contract.get("approval", {}).get("user_response_verbatim") != "批准"
        or contract.get("approval", {}).get("approved") is not True
        or contract.get("approval", {}).get(
            "prospective_before_successor_natural_scan"
        )
        is not True
    ):
        raise ValueError("A5R2 contract approval identity changed")
    predecessor = contract.get("immutable_predecessor", {})
    if (
        predecessor.get("base_commit") != BASE_COMMIT
        or predecessor.get("blocked_execution_sha") != BLOCKED_EXECUTION_SHA
        or predecessor.get("closure_sha256") != BLOCKED_CLOSURE_SHA256
        or predecessor.get("resume_allowed") is not False
        or predecessor.get("artifact_reuse_allowed") is not False
        or predecessor.get("v8_objects") != V8_OBJECT_SHA256
    ):
        raise ValueError("A5R2 immutable predecessor binding changed")
    if contract.get("scientific_invariants", {}).get(
        "population_source_sha256"
    ) != POPULATION_SOURCE_SHA256:
        raise ValueError("A5R2 frozen population-source map changed")
    for relative, expected_sha256 in POPULATION_SOURCE_SHA256.items():
        path = repo_root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != expected_sha256
        ):
            raise ValueError(f"A5R2 frozen population source changed: {relative}")
    return contract


class _FiniteFeatureSequence:
    """Synthetic finite feature wrapper accepted through one conversion path."""

    def __init__(self, values: Sequence[str]) -> None:
        self._values = list(values)

    def to_pylist(self) -> list[str]:
        return list(self._values)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._values)


class _OutcomeGuard(dict[str, Any]):
    """Fail if a purported label-free function touches an outcome field."""

    _forbidden = {
        "answer",
        "answerKey",
        "correct_answer",
        "prediction",
        "correctness",
        "beneficial",
        "harmful",
        "accuracy",
        "loss",
        "logits",
        "model_output",
    }

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._forbidden:
            raise AssertionError(f"label-free projection touched forbidden key: {key}")
        return super().get(key, default)

    def __getitem__(self, key: str) -> Any:
        if key in self._forbidden:
            raise AssertionError(f"label-free projection touched forbidden key: {key}")
        return super().__getitem__(key)


def _descriptor(index: int, *, mismatch: bool = False) -> dict[str, Any]:
    task = "ai2-arc" if index < 128 else "openbookqa" if index < 198 else "mmlu-redux"
    config = "ARC-Challenge" if task == "ai2-arc" else "main" if task == "openbookqa" else "subject"
    dataset = (
        "allenai/ai2_arc"
        if task == "ai2-arc"
        else "openbookqa"
        if task == "openbookqa"
        else "edinburgh-dawg/mmlu-redux-2.0"
    )
    expected = {
        "dataset_id": dataset,
        "dataset_config": config,
        "dataset_split": "test",
        "evaluation_question_id": index,
        "non_outcome_native_row_id": str(index),
        "feature_schema_sha256": _sha256_bytes(f"feature:{task}".encode()),
    }
    observed = dict(expected)
    if mismatch:
        observed["evaluation_question_id"] = index + 1
    return {"expected": expected, "observed": observed}


def _synthetic_example(task: str, count: int, index: int, *, feature: bool = False) -> Mapping[str, Any]:
    choices = [f"choice-{index}-{slot}" for slot in range(count)]
    labels = [str(slot + 1) if task == "ai2-arc" else chr(65 + slot) for slot in range(count)]
    if task == "ai2-arc":
        text_value: Any = _FiniteFeatureSequence(choices) if feature else choices
        label_value: Any = _FiniteFeatureSequence(labels) if feature else labels
        row: dict[str, Any] = {
            "question": f"question-{index}",
            "choices": {"text": text_value, "label": label_value},
        }
    elif task == "openbookqa":
        row = {
            "question_stem": f"question-{index}",
            "choices": {"text": choices, "label": labels},
        }
    else:
        row = {"question": f"question-{index}", "choices": choices}
    row.update({"answerKey": "B", "correctness": True, "prediction": "B"})
    return _OutcomeGuard(row)


def _record(
    *,
    index: int,
    task: str,
    count: int,
    feature: bool = False,
    mismatch: bool = False,
) -> dict[str, Any]:
    execution_sha = "a" * 40
    run_uid = "fpct-e1-a5r2-choice-cardinality-aaaaaaaa-v1"
    example = _synthetic_example(task, count, index, feature=feature)
    descriptor = _descriptor(index, mismatch=mismatch)
    projection = choice_audit_projection_v9(task, descriptor, example)
    persisted = projection["persisted"]
    content_group = str(persisted["observed_content_group_sha256"])
    sample = _sha256_bytes(f"sample:{index}".encode())
    observed_sample = sample if not mismatch else _sha256_bytes(f"other:{index}".encode())
    # A formatter-normalization case models a legacy extractor that rejected
    # the feature wrapper while the canonical view rendered byte-identically.
    legacy_count = 0 if feature else count
    prompt_sha = _sha256_bytes(
        canonical_json_bytes(
            {
                "question": choice_payload_v9(
                    task, example, enforce_task_cardinality=False
                )["question"],
                "choices": choice_payload_v9(
                    task, example, enforce_task_cardinality=False
                )["production_choices"],
            }
        )
    )
    return choice_cardinality_record_v9(
        execution_sha=execution_sha,
        run_uid=run_uid,
        group_ordinal=index + 1,
        task=task,
        content_group_sha256=content_group,
        sample_key_sha256=sample,
        observed_sample_key_sha256=observed_sample,
        descriptor=descriptor,
        projection=projection,
        raw_prompt_sha256=prompt_sha,
        canonical_prompt_sha256=prompt_sha,
        formatter_byte_parity=True,
        legacy_choice_count=legacy_count,
    )


def run_choice_cardinality_oracles(repo_root: Path = REPO_ROOT) -> dict[str, bool]:
    """Run synthetic-only variable-cardinality and fail-closed controls."""

    from script.experiment.fpct_e1_a5_prompt_provenance import _taxonomy_v9
    from script.experiment.fpct_e1_prepare_input_lock import (
        A5R2InputLockError,
        BLOCKED_RECEIPT_NAME,
        GO_RECEIPT_NAME,
        _publish_blocked_receipt,
        validate_a5_execution_identity,
    )
    _run_inherited_hash_domain_oracles()

    # Cardinality 2/3/4/5 must preserve exact natural choice geometry.  The
    # historical projection uses min(4,n), but the production payload does not.
    for count in (2, 3, 4, 5):
        example = _synthetic_example("ai2-arc", count, count)
        payload = choice_payload_v9("ai2-arc", example)
        projected = historical_projected_example_v9("ai2-arc", example)
        projected_payload = choice_payload_v9("ai2-arc", projected)
        if payload["production_choice_count"] != count:
            raise AssertionError("A5R2 altered production choice cardinality")
        if projected_payload["production_choice_count"] != min(4, count):
            raise AssertionError("A5R2 historical min-four formula changed")
        if payload["production_choices"] != [f"choice-{count}-{slot}" for slot in range(count)]:
            raise AssertionError("A5R2 padded, truncated or reordered production choices")

    # Source option labels are provenance only.  Numeric ARC labels with a
    # runtime-A answer remain valid because runtime ordinal labels are A/B/...
    numeric = choice_payload_v9("ai2-arc", _synthetic_example("ai2-arc", 4, 99))
    if numeric["source_choice_labels"] != ["1", "2", "3", "4"] or numeric[
        "runtime_ordinal_labels"
    ] != list("ABCD"):
        raise AssertionError("source/runtime option-label domains collapsed")

    rows: list[dict[str, Any]] = []
    for index in range(326):
        if index < 128:
            task = "ai2-arc"
            count = (2, 3, 4, 5)[index % 4]
            feature = index == 0
        elif index < 198:
            task, count, feature = "openbookqa", 4, False
        else:
            task, count, feature = "mmlu-redux", 4, False
        row = _record(index=index, task=task, count=count, feature=feature)
        validate_a5r2_schema_artifact(row, repo_root=repo_root)
        rows.append(row)

    ledger = {
        "relative_path": "choice_audit/choice_cardinality_audit_rows.jsonl",
        "sha256": _sha256_bytes(canonical_json_bytes(rows)),
        "bytes": len(canonical_json_bytes(rows)),
        "row_count": 326,
    }
    source_bindings = [
        {"path": BLOCKED_CLOSURE_RELATIVE.as_posix(), "sha256": BLOCKED_CLOSURE_SHA256}
    ]
    summary = summarize_choice_cardinality_audit_v9(rows)
    summary_bytes = canonical_json_bytes(summary) + b"\n"
    summary_artifact = {
        "relative_path": "choice_audit/choice_cardinality_audit_summary.json",
        "sha256": _sha256_bytes(summary_bytes),
        "bytes": len(summary_bytes),
        "row_count": 1,
    }
    e0_data_tree_binding = {
        "file_count": 51,
        "bytes": 233569,
        "e0_declared_tree_algorithm": "relative_path_nul_file_sha256_bytes_v1",
        "e0_declared_tree_sha256": (
            "f3dcf2c77e6c5f90946994488fcb86f67dcdc590510a9f469f32e86773492c73"
        ),
        "generic_asset_tree_algorithm": "canonical_json_file_manifest_v1",
        "generic_asset_tree_sha256": (
            "12f537cade1a30f6fd4e7a146c58311412f8824a6845ea4e6f7a6b5651bcb405"
        ),
        "both_domains_recomputed_before_audit_row_one": True,
        "cross_domain_comparison_detected": False,
    }
    source_snapshot_binding = {
        "receipt_path": "/synthetic/source_snapshot/.fpct_e1_source_snapshot_receipt.json",
        "receipt_file_sha256": "1" * 64,
        "receipt_sha256": "2" * 64,
        "mounted_tree_sha256": "3" * 64,
        "execution_sha": "a" * 40,
        "independently_verified_before_audit_row_one": True,
    }
    lock = build_choice_cardinality_audit_lock_v9(
        rows,
        execution_sha="a" * 40,
        run_uid="fpct-e1-a5r2-choice-cardinality-aaaaaaaa-v1",
        run_root="/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-aaaaaaaa-v1",
        ledger_artifact=ledger,
        summary_artifact=summary_artifact,
        e0_data_tree_binding=e0_data_tree_binding,
        source_snapshot_binding=source_snapshot_binding,
        source_bindings=source_bindings,
        predecessor_v8_unchanged=True,
        independent_reduction_and_hash_verified=True,
        frozen_group_sample_membership_verified=True,
        frozen_canonical_order_verified=True,
        frozen_loader_coordinates_verified=True,
    )
    validate_a5r2_schema_artifact(lock, repo_root=repo_root)
    if lock.get("status") != "A5R2_CHOICE_AUDIT_GO":
        raise AssertionError("valid synthetic population did not reach audit GO")
    if rows[0].get("root_cause_class") != "HF_FEATURE_NORMALIZATION_ONLY":
        raise AssertionError("feature-normalization control did not reach its class")

    descriptor_bad = _record(
        index=0, task="ai2-arc", count=2, mismatch=True
    )
    validate_a5r2_schema_artifact(descriptor_bad, repo_root=repo_root)
    if descriptor_bad.get("root_cause_class") != "DESCRIPTOR_INDEX_MISMATCH":
        raise AssertionError("descriptor mismatch did not use the frozen hard stop")

    # Invalid task cardinality is classified, never padded or silently dropped.
    invalid = _record(index=0, task="ai2-arc", count=1)
    validate_a5r2_schema_artifact(invalid, repo_root=repo_root)
    if invalid.get("root_cause_class") != "GENUINE_TASK_INVALID_CARDINALITY":
        raise AssertionError("invalid cardinality did not use the frozen taxonomy")
    for index, task, count in (
        (128, "openbookqa", 3),
        (198, "mmlu-redux", 5),
    ):
        task_invalid = _record(index=index, task=task, count=count)
        validate_a5r2_schema_artifact(task_invalid, repo_root=repo_root)
        if task_invalid.get("root_cause_class") != (
            "GENUINE_TASK_INVALID_CARDINALITY"
        ):
            raise AssertionError(f"{task} invalid cardinality was not classified")

        # A parseable invalid row must survive the complete 326-row reduction
        # and produce a mechanical BLOCK lock.  Merely schema-validating the
        # individual row would miss count/label inconsistencies in summary.
        blocked_rows = list(rows)
        blocked_rows[index] = task_invalid
        blocked_summary = summarize_choice_cardinality_audit_v9(blocked_rows)
        if (
            blocked_summary.get("status") != "A5R2_CHOICE_AUDIT_BLOCKED"
            or blocked_summary.get("row_count") != 326
            or blocked_summary.get("root_cause_counts", {}).get(
                "GENUINE_TASK_INVALID_CARDINALITY"
            )
            != 1
        ):
            raise AssertionError(
                f"{task} invalid row did not produce a complete mechanical BLOCK"
            )
        blocked_summary_bytes = canonical_json_bytes(blocked_summary) + b"\n"
        blocked_lock = build_choice_cardinality_audit_lock_v9(
            blocked_rows,
            execution_sha="a" * 40,
            run_uid="fpct-e1-a5r2-choice-cardinality-aaaaaaaa-v1",
            run_root="/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-aaaaaaaa-v1",
            ledger_artifact=ledger,
            summary_artifact={
                "relative_path": (
                    "choice_audit/choice_cardinality_audit_summary.json"
                ),
                "sha256": _sha256_bytes(blocked_summary_bytes),
                "bytes": len(blocked_summary_bytes),
                "row_count": 1,
            },
            e0_data_tree_binding=e0_data_tree_binding,
            source_snapshot_binding=source_snapshot_binding,
            source_bindings=source_bindings,
            predecessor_v8_unchanged=True,
            independent_reduction_and_hash_verified=True,
            frozen_group_sample_membership_verified=True,
            frozen_canonical_order_verified=True,
            frozen_loader_coordinates_verified=True,
        )
        validate_a5r2_schema_artifact(blocked_lock, repo_root=repo_root)
        if (
            blocked_lock.get("status") != "A5R2_CHOICE_AUDIT_BLOCKED"
            or blocked_lock.get("same_execution_input_lock_consumption_allowed")
            is not False
        ):
            raise AssertionError(
                f"{task} invalid population did not close input-lock consumption"
            )

    # Prove the published precedence, including combinations that cannot form
    # a trusted persisted row after a parser failure.
    precedence = _taxonomy_v9(
        task="ai2-arc",
        count=1,
        identity_status="DESCRIPTOR_INDEX_MISMATCH",
        representation_status="UNSUPPORTED",
        cardinality_status="TASK_INVALID",
        formatter_byte_parity=False,
        legacy_choice_count=0,
    )
    if precedence[0] != "DESCRIPTOR_INDEX_MISMATCH":
        raise AssertionError("descriptor mismatch lost first taxonomy precedence")
    precedence = _taxonomy_v9(
        task="openbookqa",
        count=3,
        identity_status="MATCH",
        representation_status="UNSUPPORTED",
        cardinality_status="TASK_INVALID",
        formatter_byte_parity=False,
        legacy_choice_count=0,
    )
    if precedence[0] != "UNSUPPORTED_FEATURE_REPRESENTATION":
        raise AssertionError("unsupported representation lost taxonomy precedence")

    # Mutating the lock from GO to a superficially favorable but inconsistent
    # state must fail strict schema validation.
    mutated = copy.deepcopy(lock)
    mutated["taxonomy_counts"]["DESCRIPTOR_INDEX_MISMATCH"] = 1
    try:
        validate_a5r2_schema_artifact(mutated, repo_root=repo_root)
    except ValueError:
        pass
    else:
        raise AssertionError("schema accepted a GO lock with a hard-stop taxonomy")

    # A source object containing outcomes is safe only because the projection
    # did not touch them; the guard above would have raised synchronously.
    if any(
        name in rows[0]
        for name in ("answer", "answerKey", "prediction", "correctness", "logits")
    ):
        raise AssertionError("outcome field escaped into the audit ledger")

    # Execute the real producer terminal transition: an audit-stage failure
    # must publish one strict BLOCKED receipt and no canonical GO.
    with tempfile.TemporaryDirectory(prefix="fpct-a5r2-terminal-") as temporary:
        run_root = Path(temporary) / "fpct-e1-a5r2-aaaaaaaa-v1"
        output_root = run_root / "input_lock"
        output_root.mkdir(parents=True)
        identity = {
            "execution_sha": "a" * 40,
            "run_uid": "fpct-e1-a5r2-choice-cardinality-aaaaaaaa-v1",
            "run_root": "/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-aaaaaaaa-v1",
            "source_snapshot_root": str(repo_root),
        }
        _publish_blocked_receipt(
            output_root,
            A5R2InputLockError(
                "CHOICE_AUDIT",
                "SYNTHETIC_TERMINAL_FAILURE",
                failed_group_ordinal=1,
            ),
            identity,
            repo_root=repo_root,
            e0_data_root=repo_root,
        )
        blocked_path = output_root / BLOCKED_RECEIPT_NAME
        if not blocked_path.is_file() or (output_root / GO_RECEIPT_NAME).exists():
            raise AssertionError("producer terminal failure did not publish BLOCKED-only")
        blocked = _load_json_unique(blocked_path)
        validate_a5r2_schema_artifact(blocked, repo_root=repo_root)
        if (
            blocked.get("failure_stage") != "CHOICE_AUDIT"
            or blocked.get("failed_group_ordinal") != 1
            or blocked.get("complete_choice_audit_lock_published") is not False
        ):
            raise AssertionError("producer terminal failure context changed")

    # Execute the real identity guard.  It must reject 37be before following a
    # receipt or inspecting a run root, so no historical state can be reused.
    try:
        validate_a5_execution_identity(
            execution_sha=BLOCKED_EXECUTION_SHA,
            source_snapshot_root=Path("/nonexistent/a5r2-source"),
            source_snapshot_receipt=Path("/nonexistent/a5r2-receipt"),
            run_uid="fpct-e1-a5r2-choice-cardinality-37be816a-v1",
            run_root=Path("/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-37be816a-v1"),
            output_root=Path(
                "/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-37be816a-v1/input_lock"
            ),
        )
    except ValueError as error:
        if "historical abandoned execution identity" not in str(error):
            raise AssertionError("historical identity failed for the wrong reason") from error
    else:
        raise AssertionError("historical 37be execution identity was reusable")

    return {
        "taxonomy_precedence_pass": True,
        "task_cardinality_cases_pass": True,
        "historical_min_four_formula_pass": True,
        "label_free_firewall_pass": True,
        "descriptor_mismatch_hard_stop_pass": True,
        "normalization_prompt_parity_gate_pass": True,
        "audit_go_and_blocked_mutations_pass": True,
        "v8_inheritance_boundary_pass": True,
        "fresh_uid_root_nonreuse_pass": True,
        "atomic_failure_publishes_no_go_pass": True,
    }


def _run_tests(repo_root: Path) -> tuple[int, str]:
    missing = [relative for relative in TEST_FILES if not (repo_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"A5R2 gate test closure is incomplete: {missing}")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-cov",
        "-p",
        "no:cacheprovider",
        "--basetemp=/tmp/pytest-e1-a5r2-choice-cardinality-gate",
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
        raise RuntimeError("A5R2 pytest output does not report a passed count")
    return int(matches[-1].group("count")), _sha256_bytes(output.encode())


def build_gate(*, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    _require_offline_cpu_environment()
    predecessors = _verify_predecessors(repo_root)
    _load_contract(repo_root)
    oracles = run_choice_cardinality_oracles(repo_root)
    passed, output_sha256 = _run_tests(repo_root)
    v9_artifacts = [
        _file_binding(repo_root, relative.as_posix())
        for relative in (AMENDMENT_RELATIVE, CONTRACT_RELATIVE, SCHEMA_RELATIVE)
    ]
    checks = {
        "contract_validates": True,
        "schema_self_consistent": True,
        "approval_verbatim_bound": True,
        "f782_base_bound": True,
        "37be_failure_closure_bound": True,
        "v8_four_object_immutability_pass": True,
        "inherited_v8_streaming_evidence_exact_pass": True,
        "taxonomy_precedence_pass": oracles["taxonomy_precedence_pass"],
        "task_cardinality_cases_pass": oracles["task_cardinality_cases_pass"],
        "historical_min_four_formula_pass": oracles[
            "historical_min_four_formula_pass"
        ],
        "label_free_firewall_pass": oracles["label_free_firewall_pass"],
        "descriptor_mismatch_hard_stop_pass": oracles[
            "descriptor_mismatch_hard_stop_pass"
        ],
        "normalization_prompt_parity_gate_pass": oracles[
            "normalization_prompt_parity_gate_pass"
        ],
        "audit_go_and_blocked_mutations_pass": oracles[
            "audit_go_and_blocked_mutations_pass"
        ],
        "v8_inheritance_boundary_pass": oracles[
            "v8_inheritance_boundary_pass"
        ],
        "fresh_uid_root_nonreuse_pass": oracles["fresh_uid_root_nonreuse_pass"],
        "atomic_failure_publishes_no_go_pass": oracles[
            "atomic_failure_publishes_no_go_pass"
        ],
        "natural_e0_design_accessed": False,
        "model_or_checkpoint_loaded": False,
        "model_forward_run": False,
        "gpu_cuda_or_kubernetes_used": False,
        "training": False,
        "e1_2_or_e1_3_authorized": False,
    }
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": A5R2_PROTOCOL_ID,
        "artifact_type": "a5r2_choice_cardinality_synthetic_gate",
        "status": "GO_PRE_NATURAL_A5R2_CHOICE_CARDINALITY_HARD_GATE",
        "base_commit": BASE_COMMIT,
        "blocked_execution_sha": BLOCKED_EXECUTION_SHA,
        "blocked_failure_closure_sha256": BLOCKED_CLOSURE_SHA256,
        "v8_objects": predecessors,
        "inherited_v8_streaming_evidence": _inherited_v8_streaming_evidence(
            repo_root
        ),
        "v9_artifacts": v9_artifacts,
        "tracked_files": _tracked_file_bindings(repo_root),
        "execution_tree_sha256": _execution_tree_sha256(repo_root),
        "checks": checks,
        "test_count": passed,
        "test_output_sha256": output_sha256,
    }
    value["evidence_sha256"] = _evidence_sha256(value)
    validate_a5r2_schema_artifact(value, repo_root=repo_root)
    return value


def verify_gate(gate_path: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    _require_offline_cpu_environment()
    value = _load_json_unique(gate_path)
    validate_a5r2_schema_artifact(value, repo_root=repo_root)
    if value.get("v8_objects") != _verify_predecessors(repo_root):
        raise ValueError("A5R2 predecessor map changed")
    if value.get("inherited_v8_streaming_evidence") != (
        _inherited_v8_streaming_evidence(repo_root)
    ):
        raise ValueError("A5R2 inherited v8 streaming evidence changed")
    _load_contract(repo_root)
    expected_v9 = [
        _file_binding(repo_root, relative.as_posix())
        for relative in (AMENDMENT_RELATIVE, CONTRACT_RELATIVE, SCHEMA_RELATIVE)
    ]
    if value.get("v9_artifacts") != expected_v9:
        raise ValueError("A5R2 normative artifact binding changed")
    if value.get("tracked_files") != _tracked_file_bindings(repo_root):
        raise ValueError("A5R2 tracked source/test map is stale")
    if value.get("execution_tree_sha256") != _execution_tree_sha256(repo_root):
        raise ValueError("A5R2 execution tree is stale")
    if value.get("evidence_sha256") != _evidence_sha256(value):
        raise ValueError("A5R2 evidence SHA does not recompute")
    expected_oracles = run_choice_cardinality_oracles(repo_root)
    for name, result in expected_oracles.items():
        if value.get("checks", {}).get(name) is not result:
            raise ValueError(f"A5R2 oracle no longer replays: {name}")
    return value


verify_a5_gate = verify_gate
validate_a5_schema_artifact = validate_a5r2_schema_artifact


def publish_gate(value: Mapping[str, Any], output: Path) -> None:
    validate_a5r2_schema_artifact(value, repo_root=REPO_ROOT)
    payload = canonical_json_bytes(dict(value)) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable A5R2 gate")
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
