from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

import script.analysis.fpct_e1_a5r2_choice_cardinality_gate as a5r2_gate
import script.experiment.fpct_e1_prepare_input_lock as prepare
from script.experiment.fpct_e1_a5_prompt_provenance import (
    choice_audit_projection_v9,
    choice_payload_v9,
    historical_projected_example_v9,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_SHA = "a" * 40
RUN_UID = "fpct-e1-a5r2-choice-cardinality-aaaaaaaa-v1"
SCHEMA_RUN_ROOT = "/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-aaaaaaaa-v1"


def _expected_e0_binding() -> dict[str, Any]:
    return {
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


def _snapshot_binding() -> dict[str, Any]:
    return {
        "receipt_path": "/synthetic/source_snapshot_receipt.json",
        "receipt_file_sha256": "1" * 64,
        "receipt_sha256": "2" * 64,
        "mounted_tree_sha256": "3" * 64,
        "execution_sha": EXECUTION_SHA,
        "independently_verified_before_audit_row_one": True,
    }


def _strict_tmp_aware_schema_validator(
    value: Mapping[str, Any], repo_root: Path
) -> None:
    """Run the real strict schema while substituting only its /netdisk root regex.

    Producer integration tests must write under ``tmp_path``.  The run-root
    regex is a deployment-location constraint rather than an artifact-shape
    constraint, so a deep copy receives the canonical production spelling;
    every other byte-level/schema field is validated unchanged.
    """

    candidate = copy.deepcopy(dict(value))
    if "run_root" in candidate:
        candidate["run_root"] = SCHEMA_RUN_ROOT
    a5r2_gate.validate_a5r2_schema_artifact(candidate, repo_root=REPO_ROOT)


class _SyntheticFormatter:
    """Pure formatter with the same full-choice ordinal semantics as E0."""

    dataset_name: str

    def format_example(self, example: Mapping[str, Any], use_cot: bool) -> str:
        assert use_cot is False
        payload = choice_payload_v9(
            self.dataset_name, example, enforce_task_cardinality=False
        )
        return json.dumps(
            {
                "question": payload["question"],
                "choices": [
                    [label, text]
                    for label, text in zip(
                        payload["runtime_ordinal_labels"],
                        payload["production_choices"],
                    )
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _example(task: str, index: int, count: int) -> dict[str, Any]:
    question = f"synthetic-{task}-question-{index}"
    choices = [f"synthetic-{task}-{index}-choice-{slot}" for slot in range(count)]
    labels = [chr(65 + slot) for slot in range(count)]
    if task == "ai2-arc":
        return {
            "question": question,
            "choices": {"text": choices, "label": labels},
            # These sentinels prove the producer's allowlist drops outcomes.
            "answerKey": "DO_NOT_READ",
            "correctness": "DO_NOT_READ",
        }
    if task == "openbookqa":
        return {
            "question_stem": question,
            "choices": {"text": choices, "label": labels},
            "answerKey": "DO_NOT_READ",
        }
    assert task == "mmlu-redux"
    return {
        "question": question,
        "choices": choices,
        "answer": "DO_NOT_READ",
    }


def _question_and_choices(task: str, example: Mapping[str, Any]) -> tuple[str, list[str]]:
    if task == "openbookqa":
        question = str(example["question_stem"])
    else:
        question = str(example["question"])
    raw = example["choices"]
    choices = list(raw["text"] if isinstance(raw, Mapping) else raw)
    return question, [str(value) for value in choices]


def _synthetic_population() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    examples: dict[str, dict[str, Any]] = {}
    task_counts = {"ai2-arc": 128, "openbookqa": 70, "mmlu-redux": 128}
    for task, population_count in task_counts.items():
        for index in range(population_count):
            count = (2, 3, 4, 5)[index % 4] if task == "ai2-arc" else 4
            example = _example(task, index, count)
            question, choices = _question_and_choices(task, example)
            group = prepare.canonical_content_sha256(question, choices)
            source_row_id = f"{task}-{index}"
            subject = "synthetic-mmlu-subject" if task == "mmlu-redux" else task
            sample = prepare.canonical_sample_sha256(task, subject, source_row_id)
            assert group not in records
            records[group] = {
                "task": task,
                "subject": subject,
                "source_row_id": source_row_id,
                "evaluation_question_id": index,
                "sample_sha256": sample,
            }
            examples[source_row_id] = example
    assert len(records) == 326
    return {"records": records}, examples


def _install_audit_mocks(
    monkeypatch: pytest.MonkeyPatch,
    examples: Mapping[str, Mapping[str, Any]],
    dev: Mapping[str, Any],
) -> None:
    monkeypatch.setattr(
        prepare,
        "_a5r2_pre_audit_asset_bindings",
        lambda **kwargs: (_expected_e0_binding(), _snapshot_binding()),
    )
    monkeypatch.setattr(
        prepare,
        "_a5r2_choice_audit_source_bindings",
        lambda repo_root: [
            {"path": "synthetic/frozen-source", "sha256": "4" * 64}
        ],
    )
    monkeypatch.setattr(
        prepare,
        "_load_task_example",
        lambda **kwargs: copy.deepcopy(
            examples[str(kwargs["descriptor"]["source_row_id"])]
        ),
    )
    monkeypatch.setattr(
        prepare, "_load_frozen_a5r2_dev", lambda _repo_root: copy.deepcopy(dev)
    )
    monkeypatch.setattr(
        prepare, "_validate_a5r2_schema_artifact", _strict_tmp_aware_schema_validator
    )


def _identity(run_root: Path) -> dict[str, Any]:
    return {
        "execution_sha": EXECUTION_SHA,
        "run_uid": RUN_UID,
        "run_root": str(run_root),
        "source_snapshot_root": str(REPO_ROOT),
    }


def _run_synthetic_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    dev, examples = _synthetic_population()
    run_root = tmp_path / "synthetic-a5r2-run"
    run_root.mkdir()
    e0_data_root = tmp_path / "synthetic-e0-data"
    e0_data_root.mkdir()
    _install_audit_mocks(monkeypatch, examples, dev)
    verified = prepare._run_a5r2_choice_cardinality_audit(
        repo_root=REPO_ROOT,
        e0_data_root=e0_data_root,
        dev=dev,
        execution_identity=_identity(run_root),
        a5_contract={"synthetic": True},
        formatter_class=_SyntheticFormatter,
        reference_model_config={},
    )
    return verified, dev, examples


def test_mocked_326_row_audit_publishes_three_files_and_strictly_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, _dev, _examples = _run_synthetic_audit(tmp_path, monkeypatch)
    audit_root = tmp_path / "synthetic-a5r2-run" / prepare.CHOICE_AUDIT_ROOT_NAME
    assert {path.name for path in audit_root.iterdir()} == {
        prepare.CHOICE_AUDIT_LEDGER_NAME,
        prepare.CHOICE_AUDIT_SUMMARY_NAME,
        prepare.CHOICE_AUDIT_LOCK_NAME,
    }
    assert len(verified["records"]) == 326
    assert verified["lock"]["status"] == "A5R2_CHOICE_AUDIT_GO"
    assert verified["lock"]["task_counts"] == {
        "ai2-arc": 128,
        "openbookqa": 70,
        "mmlu-redux": 128,
    }
    arc_counts = verified["lock"]["cardinality_by_task"]["ai2-arc"]
    assert {key: arc_counts[key] for key in ("two", "three", "four", "five_plus")} == {
        "two": 32,
        "three": 32,
        "four": 32,
        "five_plus": 32,
    }
    assert all(
        row["gold_or_outcome_field_accessed"] is False
        for row in verified["records"]
    )
    replay = prepare._verify_a5r2_choice_audit(
        repo_root=REPO_ROOT,
        e0_data_root=tmp_path / "synthetic-e0-data",
        execution_identity=_identity(tmp_path / "synthetic-a5r2-run"),
        a5_contract={"synthetic": True},
        require_go=True,
    )
    assert replay["lock"] == verified["lock"]


def test_audit_call_is_before_tokenizer_and_alignment_calls_in_producer_ast() -> None:
    tree = ast.parse(
        (REPO_ROOT / "script/experiment/fpct_e1_prepare_input_lock.py").read_text(
            encoding="utf-8"
        )
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_prepare_input_lock_after_identity"
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]

    def call_name(call: ast.Call) -> str:
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return ""

    audit_line = min(
        call.lineno
        for call in calls
        if call_name(call) == "_run_a5r2_choice_cardinality_audit"
    )
    forbidden_after_only = {
        "from_pretrained",
        "set_default_chat_template",
        "_build_aligner",
        "_prompt_alignment_details",
        "_verify_feature_alignment",
    }
    guarded = [call for call in calls if call_name(call) in forbidden_after_only]
    assert guarded
    assert all(call.lineno > audit_line for call in guarded)


def test_staging_publish_failure_leaves_no_final_audit_or_usable_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev, examples = _synthetic_population()
    run_root = tmp_path / "synthetic-a5r2-run"
    run_root.mkdir()
    e0_data_root = tmp_path / "synthetic-e0-data"
    e0_data_root.mkdir()
    _install_audit_mocks(monkeypatch, examples, dev)
    original_publish = prepare.publish_bytes_no_overwrite
    publish_count = 0

    def fail_summary_publish(path: Path, payload: bytes) -> str:
        nonlocal publish_count
        publish_count += 1
        if publish_count == 2:
            raise OSError("injected staged summary publish failure")
        return original_publish(path, payload)

    monkeypatch.setattr(prepare, "publish_bytes_no_overwrite", fail_summary_publish)
    with pytest.raises(OSError, match="staged summary"):
        prepare._run_a5r2_choice_cardinality_audit(
            repo_root=REPO_ROOT,
            e0_data_root=e0_data_root,
            dev=dev,
            execution_identity=_identity(run_root),
            a5_contract={"synthetic": True},
            formatter_class=_SyntheticFormatter,
            reference_model_config={},
        )
    assert not (run_root / prepare.CHOICE_AUDIT_ROOT_NAME).exists()
    assert not list(run_root.glob(".choice_audit.staging.*"))
    assert not list(run_root.rglob(prepare.CHOICE_AUDIT_LOCK_NAME))


def test_atomic_directory_publication_never_replaces_concurrent_empty_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev, examples = _synthetic_population()
    run_root = tmp_path / "synthetic-a5r2-run"
    run_root.mkdir()
    e0_data_root = tmp_path / "synthetic-e0-data"
    e0_data_root.mkdir()
    _install_audit_mocks(monkeypatch, examples, dev)
    original_rename = prepare._rename_directory_no_replace

    def inject_competing_target(source: Path, destination: Path) -> None:
        destination.mkdir()
        original_rename(source, destination)

    monkeypatch.setattr(
        prepare, "_rename_directory_no_replace", inject_competing_target
    )
    with pytest.raises(FileExistsError):
        prepare._run_a5r2_choice_cardinality_audit(
            repo_root=REPO_ROOT,
            e0_data_root=e0_data_root,
            dev=dev,
            execution_identity=_identity(run_root),
            a5_contract={"synthetic": True},
            formatter_class=_SyntheticFormatter,
            reference_model_config={},
        )
    final_root = run_root / prepare.CHOICE_AUDIT_ROOT_NAME
    assert final_root.is_dir()
    assert list(final_root.iterdir()) == []
    assert not list(run_root.glob(".choice_audit.staging.*"))
    assert not list(run_root.rglob(prepare.CHOICE_AUDIT_LOCK_NAME))


def test_pre_audit_binding_recomputes_exact_receipt_file_sha_before_row_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    e0_data_root = tmp_path / "synthetic-e0-data"
    e0_data_root.mkdir()
    snapshot = tmp_path / "source_snapshot"
    snapshot.mkdir()
    receipt = snapshot / ".fpct_e1_source_snapshot_receipt.json"
    receipt.write_text('{"synthetic":true}\n', encoding="utf-8")
    expected_tree = {
        "file_count": 51,
        "bytes": 233569,
        "e0_declared_tree_algorithm": "relative_path_nul_file_sha256_bytes_v1",
        "e0_declared_tree_sha256": "1" * 64,
        "generic_asset_tree_algorithm": "canonical_json_file_manifest_v1",
        "generic_asset_tree_sha256": "2" * 64,
    }
    monkeypatch.setattr(
        prepare, "_e0_data_asset_tree", lambda _root: {"synthetic": True}
    )
    monkeypatch.setattr(
        prepare,
        "_e0_data_hash_domain_projection",
        lambda _record: dict(expected_tree),
    )
    identity = {
        "execution_sha": "a" * 40,
        "source_snapshot_root": str(snapshot),
        "source_snapshot_receipt": {
            "path": str(receipt),
            "file_sha256": "0" * 64,
            "verification": {"synthetic": True},
        },
    }
    with pytest.raises(
        prepare.A5R2InputLockError,
        match="PRE_AUDIT_SOURCE_RECEIPT_FILE_SHA_MISMATCH",
    ):
        prepare._a5r2_pre_audit_asset_bindings(
            e0_data_root=e0_data_root,
            contract={
                "asset_identity": {
                    "materialized_e0_dev_data_tree": expected_tree
                }
            },
            execution_identity=identity,
        )


def test_parse_failure_is_wrapped_with_exact_group_ordinal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev, examples = _synthetic_population()
    ordered = [
        (group, descriptor)
        for task in prepare.TASKS
        for group, descriptor in sorted(dev["records"].items())
        if descriptor["task"] == task
    ]
    failed_ordinal = 17
    failed_source_row_id = ordered[failed_ordinal - 1][1]["source_row_id"]
    examples[failed_source_row_id] = {
        "question": "malformed choices",
        "choices": "not-a-choice-sequence",
        "answerKey": "DO_NOT_READ",
    }
    run_root = tmp_path / "synthetic-a5r2-run"
    run_root.mkdir()
    e0_data_root = tmp_path / "synthetic-e0-data"
    e0_data_root.mkdir()
    _install_audit_mocks(monkeypatch, examples, dev)
    with pytest.raises(prepare.A5R2InputLockError) as caught:
        prepare._run_a5r2_choice_cardinality_audit(
            repo_root=REPO_ROOT,
            e0_data_root=e0_data_root,
            dev=dev,
            execution_identity=_identity(run_root),
            a5_contract={"synthetic": True},
            formatter_class=_SyntheticFormatter,
            reference_model_config={},
        )
    assert caught.value.failure_stage == "CHOICE_AUDIT"
    assert caught.value.failed_group_ordinal == failed_ordinal
    assert caught.value.failure_code
    assert not (run_root / prepare.CHOICE_AUDIT_ROOT_NAME).exists()


def test_complete_invalid_population_publishes_strict_terminal_blocked_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev, examples = _synthetic_population()
    ordered = [
        (group, descriptor)
        for task in prepare.TASKS
        for group, descriptor in sorted(dev["records"].items())
        if descriptor["task"] == task
    ]
    invalid_group, invalid_descriptor = next(
        (group, descriptor)
        for group, descriptor in ordered
        if descriptor["task"] == "openbookqa"
    )
    invalid_example = _example(
        "openbookqa", int(invalid_descriptor["evaluation_question_id"]), 3
    )
    examples[str(invalid_descriptor["source_row_id"])] = invalid_example
    question, choices = _question_and_choices("openbookqa", invalid_example)
    replacement_group = prepare.canonical_content_sha256(question, choices)
    dev["records"].pop(invalid_group)
    dev["records"][replacement_group] = invalid_descriptor
    ordered = [
        (group, descriptor)
        for task in prepare.TASKS
        for group, descriptor in sorted(dev["records"].items())
        if descriptor["task"] == task
    ]
    invalid_ordinal = next(
        ordinal
        for ordinal, (_group, descriptor) in enumerate(ordered, start=1)
        if descriptor["source_row_id"] == invalid_descriptor["source_row_id"]
    )
    run_root = tmp_path / "synthetic-a5r2-run"
    output_root = run_root / "input_lock"
    output_root.mkdir(parents=True)
    e0_data_root = tmp_path / "synthetic-e0-data"
    e0_data_root.mkdir()
    _install_audit_mocks(monkeypatch, examples, dev)
    identity = _identity(run_root)
    with pytest.raises(prepare.A5R2InputLockError) as caught:
        prepare._run_a5r2_choice_cardinality_audit(
            repo_root=REPO_ROOT,
            e0_data_root=e0_data_root,
            dev=dev,
            execution_identity=identity,
            a5_contract={"synthetic": True},
            formatter_class=_SyntheticFormatter,
            reference_model_config={},
        )
    error = caught.value
    assert error.failure_stage == "CHOICE_AUDIT_COMPLETE_MECHANICAL_BLOCK"
    assert error.failed_group_ordinal == invalid_ordinal
    audit_lock = json.loads(
        (
            run_root
            / prepare.CHOICE_AUDIT_ROOT_NAME
            / prepare.CHOICE_AUDIT_LOCK_NAME
        ).read_text(encoding="utf-8")
    )
    assert audit_lock["status"] == "A5R2_CHOICE_AUDIT_BLOCKED"

    prepare._publish_blocked_receipt(
        output_root,
        error,
        identity,
        repo_root=REPO_ROOT,
        e0_data_root=e0_data_root,
    )
    blocked = json.loads(
        (output_root / prepare.BLOCKED_RECEIPT_NAME).read_text(encoding="utf-8")
    )
    assert blocked["failure_stage"] == (
        "CHOICE_AUDIT_COMPLETE_MECHANICAL_BLOCK"
    )
    assert blocked["failed_group_ordinal"] == invalid_ordinal
    assert blocked["choice_audit_lock"] is not None
    assert blocked["complete_choice_audit_lock_published"] is True
    assert not (output_root / prepare.GO_RECEIPT_NAME).exists()
    _strict_tmp_aware_schema_validator(blocked, REPO_ROOT)


def test_exact_abandoned_37be_execution_identity_is_rejected(tmp_path: Path) -> None:
    old_sha = "37be816ad611b8b0d916bd98c840c5f31efe2b50"
    with pytest.raises(ValueError, match="historical abandoned execution"):
        prepare.validate_a5_execution_identity(
            execution_sha=old_sha,
            source_snapshot_root=tmp_path / "source_snapshot",
            source_snapshot_receipt=tmp_path / "source_snapshot/receipt.json",
            run_uid="fpct-e1-a5r1-hash-domains-37be816a-v1",
            run_root=tmp_path / "fpct-e1-a5r1-37be816a-v1",
            output_root=tmp_path / "input_lock",
        )


class _FeatureSequence:
    def __init__(self, values: list[str]):
        self._values = values

    def to_pylist(self) -> list[str]:
        return list(self._values)


def test_feature_backed_audit_projection_flows_to_historical_projection() -> None:
    example = _example("ai2-arc", 0, 5)
    raw_choices = example["choices"]
    assert isinstance(raw_choices, Mapping)
    example["choices"] = {
        "text": _FeatureSequence(list(raw_choices["text"])),
        "label": _FeatureSequence(list(raw_choices["label"])),
    }
    descriptor_fields = {
        "dataset_id": "allenai/ai2_arc",
        "dataset_config": "ARC-Challenge",
        "dataset_split": "test",
        "evaluation_question_id": 0,
        "non_outcome_native_row_id": None,
        "feature_schema_sha256": "5" * 64,
    }
    projection = choice_audit_projection_v9(
        "ai2-arc",
        {"expected": descriptor_fields, "observed": dict(descriptor_fields)},
        example,
    )
    assert projection["persisted"]["representation_status"] == (
        "HF_FEATURE_NORMALIZATION_REQUIRED"
    )
    assert projection["persisted"]["production_choice_count"] == 5
    historical = historical_projected_example_v9("ai2-arc", example)
    assert historical["choices"]["text"] == [
        f"synthetic-ai2-arc-0-choice-{slot}" for slot in range(4)
    ]
    assert len(choice_payload_v9("ai2-arc", example)["production_choices"]) == 5


def test_public_prepare_failure_reaches_real_blocked_transition_without_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "input_lock"
    output.mkdir()
    identity = {
        "execution_sha": EXECUTION_SHA,
        "run_uid": RUN_UID,
        "run_root": str(tmp_path),
        "source_snapshot_root": str(REPO_ROOT),
    }
    monkeypatch.setattr(
        prepare, "validate_a5_execution_identity", lambda **kwargs: identity
    )
    monkeypatch.setattr(
        prepare,
        "_prepare_input_lock_after_identity",
        lambda **kwargs: (_ for _ in ()).throw(
            prepare.A5R2InputLockError(
                "CHOICE_AUDIT", "INJECTED_PARSE_FAILURE", failed_group_ordinal=9
            )
        ),
    )
    monkeypatch.setattr(prepare, "_load_a5_prompt_contract", lambda repo_root: {})
    monkeypatch.setattr(
        prepare,
        "_verify_a5r2_choice_audit",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError("no audit lock")),
    )
    monkeypatch.setattr(
        prepare, "_validate_a5r2_schema_artifact", _strict_tmp_aware_schema_validator
    )
    with pytest.raises(prepare.A5R2InputLockError, match="INJECTED_PARSE_FAILURE"):
        prepare.prepare_input_lock(
            repo_root=REPO_ROOT,
            e0_data_root=tmp_path / "synthetic-e0-data",
            output_sidecar=output / "sidecar.pt",
            output_manifest=output / "manifest.json",
            execution_sha=EXECUTION_SHA,
            source_snapshot_root=REPO_ROOT,
            source_snapshot_receipt=REPO_ROOT / "unused-receipt.json",
            run_uid=RUN_UID,
            run_root=tmp_path,
            _test_only_sealed_execution=(
                prepare._verified_test_sealed_prepare_sentinel(
                    REPO_ROOT, EXECUTION_SHA
                )
            ),
        )
    blocked_path = output / prepare.BLOCKED_RECEIPT_NAME
    assert blocked_path.is_file()
    blocked = json.loads(blocked_path.read_text(encoding="utf-8"))
    assert blocked["status"] == "A5R2_INPUT_LOCK_BLOCKED"
    assert blocked["failure_stage"] == "CHOICE_AUDIT"
    assert blocked["failure_code"] == "INJECTED_PARSE_FAILURE"
    assert blocked["failed_group_ordinal"] == 9
    assert blocked["complete_choice_audit_lock_published"] is False
    assert not (output / prepare.GO_RECEIPT_NAME).exists()


@pytest.mark.parametrize("artifact", ["ledger", "summary", "lock"])
def test_independent_verifier_rejects_each_tampered_audit_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    _verified, _dev, _examples = _run_synthetic_audit(tmp_path, monkeypatch)
    run_root = tmp_path / "synthetic-a5r2-run"
    audit_root = run_root / prepare.CHOICE_AUDIT_ROOT_NAME
    paths = {
        "ledger": audit_root / prepare.CHOICE_AUDIT_LEDGER_NAME,
        "summary": audit_root / prepare.CHOICE_AUDIT_SUMMARY_NAME,
        "lock": audit_root / prepare.CHOICE_AUDIT_LOCK_NAME,
    }
    path = paths[artifact]
    original = path.read_bytes()
    if artifact == "ledger":
        lines = original.splitlines(keepends=True)
        first = json.loads(lines[0])
        first["question_nonempty"] = False
        lines[0] = prepare.canonical_json_bytes(first)
        path.write_bytes(b"".join(lines))
    else:
        value = json.loads(original)
        if artifact == "summary":
            value["row_count"] = 325
        else:
            value["same_execution_input_lock_consumption_allowed"] = False
        path.write_bytes(prepare.canonical_json_bytes(value))
    with pytest.raises((RuntimeError, ValueError)):
        prepare._verify_a5r2_choice_audit(
            repo_root=REPO_ROOT,
            e0_data_root=tmp_path / "synthetic-e0-data",
            execution_identity=_identity(run_root),
            a5_contract={"synthetic": True},
            require_go=True,
        )


def test_independent_verifier_rejects_reordered_ledger_after_full_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, dev, _examples = _run_synthetic_audit(tmp_path, monkeypatch)
    run_root = tmp_path / "synthetic-a5r2-run"
    audit_root = run_root / prepare.CHOICE_AUDIT_ROOT_NAME
    records = copy.deepcopy(verified["records"])
    records[0], records[1] = records[1], records[0]
    for ordinal, record in enumerate(records, start=1):
        record["group_ordinal"] = ordinal
    ledger_path = audit_root / prepare.CHOICE_AUDIT_LEDGER_NAME
    summary_path = audit_root / prepare.CHOICE_AUDIT_SUMMARY_NAME
    lock_path = audit_root / prepare.CHOICE_AUDIT_LOCK_NAME
    ledger_path.write_bytes(
        b"".join(prepare.canonical_json_bytes(record) for record in records)
    )
    summary = prepare.summarize_choice_cardinality_audit_v9(records)
    summary_path.write_bytes(prepare.canonical_json_bytes(summary))
    frozen_identity = prepare._a5r2_frozen_identity_checks(records, dev)
    assert frozen_identity["frozen_group_sample_membership_verified"] is True
    assert frozen_identity["frozen_canonical_order_verified"] is False
    old_lock = verified["lock"]
    rebuilt = prepare.build_choice_cardinality_audit_lock_v9(
        records,
        execution_sha=EXECUTION_SHA,
        run_uid=RUN_UID,
        run_root=str(run_root),
        ledger_artifact=prepare._choice_audit_artifact_descriptor(
            ledger_path, 326
        ),
        summary_artifact=prepare._choice_audit_artifact_descriptor(
            summary_path, 1
        ),
        e0_data_tree_binding=old_lock["e0_data_tree_binding"],
        source_snapshot_binding=old_lock["source_snapshot_binding"],
        source_bindings=old_lock["source_bindings"],
        predecessor_v8_unchanged=True,
        independent_reduction_and_hash_verified=True,
        **frozen_identity,
    )
    lock_path.write_bytes(prepare.canonical_json_bytes(rebuilt))
    with pytest.raises(RuntimeError, match="frozen membership/order"):
        prepare._verify_a5r2_choice_audit(
            repo_root=REPO_ROOT,
            e0_data_root=tmp_path / "synthetic-e0-data",
            execution_identity=_identity(run_root),
            a5_contract={"synthetic": True},
            require_go=False,
        )
