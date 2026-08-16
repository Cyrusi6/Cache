from __future__ import annotations

import ast
import copy
import hashlib
import inspect
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


def _strict_tmp_aware_a5r3_validator(
    value: Mapping[str, Any], execution_identity: Mapping[str, Any]
) -> None:
    candidate = copy.deepcopy(dict(value))
    candidate["run_root"] = SCHEMA_RUN_ROOT
    a5r3_gate = __import__(
        "script.analysis.fpct_e1_a5r3_portable_publication_gate",
        fromlist=["validate_a5r3_schema_artifact"],
    )
    a5r3_gate.validate_a5r3_schema_artifact(candidate, repo_root=REPO_ROOT)


@pytest.fixture(autouse=True)
def _validate_a5r3_tmp_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prepare,
        "_validate_a5r3_publication_schema_artifact",
        _strict_tmp_aware_a5r3_validator,
    )


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
    claim_path = (
        tmp_path
        / "synthetic-a5r2-run"
        / prepare.CHOICE_AUDIT_PUBLICATION_CLAIM_NAME
    )
    receipt_path = (
        tmp_path
        / "synthetic-a5r2-run"
        / prepare.CHOICE_AUDIT_PUBLICATION_RECEIPT_NAME
    )
    assert claim_path.is_file()
    assert receipt_path.is_file()
    assert claim_path.stat().st_mode & 0o777 == 0o600
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / "synthetic-a5r2-run" / ".choice_audit.staging").exists()
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert prepare.canonical_json_bytes(claim) == claim_path.read_bytes()
    assert prepare.canonical_json_bytes(receipt) == receipt_path.read_bytes()
    assert receipt["claim_sha256"] == prepare.hashlib.sha256(
        claim_path.read_bytes()
    ).hexdigest()
    assert receipt["status"] == "A5R3_CHOICE_AUDIT_PUBLICATION_GO"
    assert verified["publication"] == receipt
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
    assert (run_root / prepare.CHOICE_AUDIT_PUBLICATION_CLAIM_NAME).is_file()
    assert (run_root / ".choice_audit.staging").is_dir()
    assert not (run_root / prepare.CHOICE_AUDIT_PUBLICATION_RECEIPT_NAME).exists()
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
    original_assert = prepare._assert_published_choice_audit_bytes
    injected = False

    def inject_competing_target(path: Path, **kwargs: Any) -> None:
        nonlocal injected
        original_assert(path, **kwargs)
        if not injected and path.name == ".choice_audit.staging":
            injected = True
            (run_root / prepare.CHOICE_AUDIT_ROOT_NAME).mkdir()

    monkeypatch.setattr(
        prepare, "_assert_published_choice_audit_bytes", inject_competing_target
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
    assert (run_root / ".choice_audit.staging").is_dir()
    assert (run_root / prepare.CHOICE_AUDIT_PUBLICATION_CLAIM_NAME).is_file()
    assert not (run_root / prepare.CHOICE_AUDIT_PUBLICATION_RECEIPT_NAME).exists()
    assert not list(final_root.rglob(prepare.CHOICE_AUDIT_LOCK_NAME))


@pytest.mark.parametrize("preexisting", ["claim", "staging", "final", "receipt"])
def test_preexisting_publication_state_is_terminal_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting: str,
) -> None:
    dev, examples = _synthetic_population()
    run_root = tmp_path / "synthetic-a5r2-run"
    run_root.mkdir()
    e0_data_root = tmp_path / "synthetic-e0-data"
    e0_data_root.mkdir()
    _install_audit_mocks(monkeypatch, examples, dev)
    paths = {
        "claim": run_root / prepare.CHOICE_AUDIT_PUBLICATION_CLAIM_NAME,
        "staging": run_root / ".choice_audit.staging",
        "final": run_root / prepare.CHOICE_AUDIT_ROOT_NAME,
        "receipt": run_root / prepare.CHOICE_AUDIT_PUBLICATION_RECEIPT_NAME,
    }
    path = paths[preexisting]
    if preexisting in {"staging", "final"}:
        path.mkdir()
    else:
        path.write_bytes(b"preexisting-owner-state\n")
    before = {
        child.name: (child.lstat().st_mode, child.read_bytes() if child.is_file() else b"")
        for child in run_root.iterdir()
    }
    with pytest.raises(prepare.A5R3PublicationOwnershipError):
        prepare._run_a5r2_choice_cardinality_audit(
            repo_root=REPO_ROOT,
            e0_data_root=e0_data_root,
            dev=dev,
            execution_identity=_identity(run_root),
            a5_contract={"synthetic": True},
            formatter_class=_SyntheticFormatter,
            reference_model_config={},
        )
    after = {
        child.name: (child.lstat().st_mode, child.read_bytes() if child.is_file() else b"")
        for child in run_root.iterdir()
    }
    assert after == before


def test_claim_symlink_is_terminal_and_target_is_not_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev, examples = _synthetic_population()
    run_root = tmp_path / "synthetic-a5r2-run"
    run_root.mkdir()
    e0_data_root = tmp_path / "synthetic-e0-data"
    e0_data_root.mkdir()
    target = tmp_path / "outside-claim-target"
    target.write_bytes(b"outside\n")
    claim_path = run_root / prepare.CHOICE_AUDIT_PUBLICATION_CLAIM_NAME
    claim_path.symlink_to(target)
    _install_audit_mocks(monkeypatch, examples, dev)
    with pytest.raises(prepare.A5R3PublicationOwnershipError):
        prepare._run_a5r2_choice_cardinality_audit(
            repo_root=REPO_ROOT,
            e0_data_root=e0_data_root,
            dev=dev,
            execution_identity=_identity(run_root),
            a5_contract={"synthetic": True},
            formatter_class=_SyntheticFormatter,
            reference_model_config={},
        )
    assert claim_path.is_symlink()
    assert target.read_bytes() == b"outside\n"


def test_claim_owner_token_rejects_tamper_and_replacement(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "synthetic-a5r2-run"
    run_root.mkdir()
    identity = _identity(run_root)
    staging_root = run_root / ".choice_audit.staging"
    audit_root = run_root / prepare.CHOICE_AUDIT_ROOT_NAME
    _claim, payload, owner_token = prepare._acquire_choice_audit_publication_claim(
        execution_identity=identity,
        staging_root=staging_root,
        audit_root=audit_root,
    )
    claim_path = run_root / prepare.CHOICE_AUDIT_PUBLICATION_CLAIM_NAME
    claim_path.write_bytes(prepare.canonical_json_bytes({"tampered": True}))
    claim_path.chmod(0o600)
    with pytest.raises(RuntimeError, match="bytes changed"):
        prepare._verify_owner_publication_file(
            claim_path,
            payload,
            role="publication claim",
            owner_token=owner_token,
        )
    claim_path.unlink()
    replacement = tmp_path / "replacement-claim"
    replacement.write_bytes(payload)
    replacement.chmod(0o600)
    claim_path.symlink_to(replacement)
    with pytest.raises(RuntimeError, match="absent, replaced, or unsafe"):
        prepare._verify_owner_publication_file(
            claim_path,
            payload,
            role="publication claim",
            owner_token=owner_token,
        )


def test_competing_claim_loser_mutates_nothing(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "synthetic-a5r2-run"
    run_root.mkdir()
    identity = _identity(run_root)
    staging_root = run_root / ".choice_audit.staging"
    audit_root = run_root / prepare.CHOICE_AUDIT_ROOT_NAME
    prepare._acquire_choice_audit_publication_claim(
        execution_identity=identity,
        staging_root=staging_root,
        audit_root=audit_root,
    )
    before = sorted(
        (path.name, path.lstat().st_mode, path.read_bytes() if path.is_file() else b"")
        for path in run_root.iterdir()
    )
    with pytest.raises(prepare.A5R3PublicationOwnershipError):
        prepare._acquire_choice_audit_publication_claim(
            execution_identity=identity,
            staging_root=staging_root,
            audit_root=audit_root,
        )
    after = sorted(
        (path.name, path.lstat().st_mode, path.read_bytes() if path.is_file() else b"")
        for path in run_root.iterdir()
    )
    assert after == before


def test_group_or_other_writable_run_root_is_rejected_before_claim(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "synthetic-a5r2-run"
    run_root.mkdir(mode=0o700)
    run_root.chmod(0o722)
    with pytest.raises(RuntimeError, match="group/other writable"):
        prepare._acquire_choice_audit_publication_claim(
            execution_identity=_identity(run_root),
            staging_root=run_root / ".choice_audit.staging",
            audit_root=run_root / prepare.CHOICE_AUDIT_ROOT_NAME,
        )
    assert list(run_root.iterdir()) == []


def test_plain_rename_failure_retains_claim_and_staging_tombstones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev, examples = _synthetic_population()
    run_root = tmp_path / "synthetic-a5r2-run"
    run_root.mkdir()
    e0_data_root = tmp_path / "synthetic-e0-data"
    e0_data_root.mkdir()
    _install_audit_mocks(monkeypatch, examples, dev)
    monkeypatch.setattr(
        prepare.os,
        "rename",
        lambda _source, _destination: (_ for _ in ()).throw(
            OSError("injected portable rename failure")
        ),
    )
    with pytest.raises(OSError, match="portable rename failure"):
        prepare._run_a5r2_choice_cardinality_audit(
            repo_root=REPO_ROOT,
            e0_data_root=e0_data_root,
            dev=dev,
            execution_identity=_identity(run_root),
            a5_contract={"synthetic": True},
            formatter_class=_SyntheticFormatter,
            reference_model_config={},
        )
    assert (run_root / prepare.CHOICE_AUDIT_PUBLICATION_CLAIM_NAME).is_file()
    assert (run_root / ".choice_audit.staging").is_dir()
    assert not (run_root / prepare.CHOICE_AUDIT_ROOT_NAME).exists()
    assert not (run_root / prepare.CHOICE_AUDIT_PUBLICATION_RECEIPT_NAME).exists()


def test_post_rename_pre_receipt_crash_is_terminal_and_never_consumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev, examples = _synthetic_population()
    run_root = tmp_path / "synthetic-a5r2-run"
    run_root.mkdir()
    e0_data_root = tmp_path / "synthetic-e0-data"
    e0_data_root.mkdir()
    _install_audit_mocks(monkeypatch, examples, dev)
    original_create = prepare._create_exclusive_owner_publication_file

    def crash_before_receipt(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("role") == "publication receipt":
            raise KeyboardInterrupt("injected post-rename crash")
        return original_create(*args, **kwargs)

    monkeypatch.setattr(
        prepare, "_create_exclusive_owner_publication_file", crash_before_receipt
    )
    with pytest.raises(KeyboardInterrupt, match="post-rename crash"):
        prepare._run_a5r2_choice_cardinality_audit(
            repo_root=REPO_ROOT,
            e0_data_root=e0_data_root,
            dev=dev,
            execution_identity=_identity(run_root),
            a5_contract={"synthetic": True},
            formatter_class=_SyntheticFormatter,
            reference_model_config={},
        )
    assert (run_root / prepare.CHOICE_AUDIT_PUBLICATION_CLAIM_NAME).is_file()
    assert (run_root / prepare.CHOICE_AUDIT_ROOT_NAME).is_dir()
    assert not (run_root / ".choice_audit.staging").exists()
    assert not (run_root / prepare.CHOICE_AUDIT_PUBLICATION_RECEIPT_NAME).exists()
    with pytest.raises(RuntimeError, match="receipt is absent, replaced, or unsafe"):
        prepare._verify_a5r2_choice_audit(
            repo_root=REPO_ROOT,
            e0_data_root=e0_data_root,
            execution_identity=_identity(run_root),
            a5_contract={"synthetic": True},
            require_go=False,
        )
    with pytest.raises(prepare.A5R3PublicationOwnershipError):
        prepare._run_a5r2_choice_cardinality_audit(
            repo_root=REPO_ROOT,
            e0_data_root=e0_data_root,
            dev=dev,
            execution_identity=_identity(run_root),
            a5_contract={"synthetic": True},
            formatter_class=_SyntheticFormatter,
            reference_model_config={},
        )


def test_receipt_parent_fsync_failure_poison_is_not_consumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev, examples = _synthetic_population()
    run_root = tmp_path / "synthetic-a5r2-run"
    run_root.mkdir()
    e0_data_root = tmp_path / "synthetic-e0-data"
    e0_data_root.mkdir()
    _install_audit_mocks(monkeypatch, examples, dev)
    original_fsync_directory = prepare._fsync_directory
    failed = False

    def fail_receipt_parent_fsync(path: Path) -> None:
        nonlocal failed
        receipt = run_root / prepare.CHOICE_AUDIT_PUBLICATION_RECEIPT_NAME
        if path == run_root and receipt.exists() and not failed:
            failed = True
            raise OSError("injected receipt parent fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(prepare, "_fsync_directory", fail_receipt_parent_fsync)
    with pytest.raises(OSError, match="receipt parent fsync"):
        prepare._run_a5r2_choice_cardinality_audit(
            repo_root=REPO_ROOT,
            e0_data_root=e0_data_root,
            dev=dev,
            execution_identity=_identity(run_root),
            a5_contract={"synthetic": True},
            formatter_class=_SyntheticFormatter,
            reference_model_config={},
        )
    receipt = run_root / prepare.CHOICE_AUDIT_PUBLICATION_RECEIPT_NAME
    assert receipt.lstat().st_mode & 0o777 == 0
    assert (run_root / prepare.CHOICE_AUDIT_ROOT_NAME).is_dir()
    assert (run_root / prepare.CHOICE_AUDIT_PUBLICATION_CLAIM_NAME).is_file()
    with pytest.raises(RuntimeError, match="receipt is absent, replaced, or unsafe"):
        prepare._verify_a5r2_choice_audit(
            repo_root=REPO_ROOT,
            e0_data_root=e0_data_root,
            execution_identity=_identity(run_root),
            a5_contract={"synthetic": True},
            require_go=False,
        )


def test_runtime_has_no_renameat2_dependency() -> None:
    source = (
        REPO_ROOT / "script/experiment/fpct_e1_prepare_input_lock.py"
    ).read_text(encoding="utf-8")
    assert "renameat2" not in source
    assert "RENAME_NOREPLACE" not in source


def test_claim_and_receipt_strictly_validate_as_v10_publication_envelopes() -> None:
    from script.analysis.fpct_e1_a5r3_portable_publication_gate import (
        validate_a5r3_schema_artifact,
    )
    run_root = Path(SCHEMA_RUN_ROOT)
    identity = {
        "execution_sha": EXECUTION_SHA,
        "run_uid": RUN_UID,
        "run_root": str(run_root),
        "source_snapshot_root": str(REPO_ROOT),
    }
    staging_root = run_root / ".choice_audit.staging"
    audit_root = run_root / prepare.CHOICE_AUDIT_ROOT_NAME
    claim = prepare._choice_audit_publication_claim_payload(
        execution_identity=identity,
        staging_root=staging_root,
        audit_root=audit_root,
    )
    validate_a5r3_schema_artifact(claim, repo_root=REPO_ROOT)
    descriptor = {
        "relative_path": "choice_audit/synthetic.json",
        "sha256": "9" * 64,
        "bytes": 1,
        "row_count": 1,
    }
    receipt = prepare._choice_audit_publication_receipt_payload(
        execution_identity=identity,
        claim_sha256=prepare.hashlib.sha256(
            prepare.canonical_json_bytes(claim)
        ).hexdigest(),
        staging_root=staging_root,
        audit_root=audit_root,
        ledger_artifact=descriptor,
        summary_artifact=descriptor,
        lock_artifact=descriptor,
    )
    validate_a5r3_schema_artifact(receipt, repo_root=REPO_ROOT)


def _complete_v9_input_manifest(
    choice_binding: Mapping[str, Any],
) -> dict[str, Any]:
    hard_checks = {
        name: True
        for name in (
            "historical_min4_projection_anchor_unchanged",
            "e0_design_membership_unchanged",
            "renderer_source_identity_attested",
            "production_renderer_exactly_attested",
            "production_data_tree_exactly_attested",
            "all_326_groups_resolved",
            "all_task_cardinalities_valid",
            "all_gold_answers_within_runtime_choices",
            "all_prompt_differences_classified",
            "historical_to_runtime_mapping_complete",
            "production_prompt_replay_equal",
            "production_alignment_replay_equal",
            "source_choice_order_preserved",
            "label_free_runtime_row_hashes_complete",
            "choice_audit_lock_bound_and_unchanged",
            "choice_corrections_match_locked_taxonomy",
            "logical_row_coverage_exact",
            "streaming_semantic_replay_equal",
            "bounded_peak_rss",
        )
    }
    hash_checks = {
        "e0_declared_domain_present": True,
        "e0_declared_algorithm_exact": True,
        "e0_declared_sha_matches_frozen_e0": True,
        "a5_generic_domain_present": True,
        "a5_generic_algorithm_exact": True,
        "a5_generic_sha_matches_same_domain_predecessor": True,
        "a5_generic_before_equals_after": True,
        "completed_verifier_recomputes_both_domains": True,
        "cross_domain_comparison_detected": False,
        "old_v7_artifact_modified": False,
        "blocked_execution_artifact_reused": False,
    }
    firewall = {
        "old_execution_artifact_reused": False,
        "model_instantiated": False,
        "model_or_checkpoint_loaded": False,
        "model_forward_run": False,
        "gpu_or_cuda_used": False,
        "kubernetes_used": False,
        "training": False,
        "e1_pilot_consumed": False,
        "confirmatory_consumed": False,
        "e1_2_or_e1_3_authorized": False,
    }
    sha = "7" * 64
    return {
        "schema_version": 9,
        "protocol_id": "fpct_e1_mechanism_audit_v9_a5r2_choice_cardinality",
        "input_lock_protocol_id": "fpct_e1_e0_design_input_lock_v5_a5r2_choice_cardinality",
        "artifact_type": "a5r2_input_lock_manifest",
        "status": "A5R2_INPUT_LOCK_GO_NO_MODEL_OUTPUT",
        "split_role": "e0_design",
        "execution": {
            "execution_sha": EXECUTION_SHA,
            "run_uid": RUN_UID,
            "run_root": SCHEMA_RUN_ROOT,
            "source_snapshot_receipt_sha256": sha,
            "source_snapshot_tree_sha256": sha,
        },
        "task_counts": {"ai2-arc": 128, "openbookqa": 70, "mmlu-redux": 128},
        "inherited_v8_contract_binding": prepare._inherited_v8_contract_binding(),
        "choice_audit": dict(choice_binding),
        "provenance": {
            "renderer_source_identity_sha256": sha,
            "prompt_config_identity_sha256": sha,
            "runtime_prompt_assets_sha256": sha,
            "e0_declared_tree_algorithm": "relative_path_nul_file_sha256_bytes_v1",
            "e0_declared_tree_sha256": "f3dcf2c77e6c5f90946994488fcb86f67dcdc590510a9f469f32e86773492c73",
            "generic_asset_tree_algorithm": "canonical_json_file_manifest_v1",
            "generic_asset_tree_sha256": "12f537cade1a30f6fd4e7a146c58311412f8824a6845ea4e6f7a6b5651bcb405",
            "input_assets_before_sha256": sha,
            "input_assets_after_sha256": sha,
            "input_assets_unchanged": True,
        },
        "census": {
            "manifest": {"path": "/synthetic/census.json", "bytes": 1, "sha256": sha},
            "records": {"path": "/synthetic/census.jsonl", "bytes": 1, "sha256": sha},
            "record_count": 326,
            "canonical_semantic_stream_sha256": sha,
            "choice_audit_lock_sha256": choice_binding["lock"]["sha256"],
        },
        "sidecar": {
            "contract_version": 5,
            "path": "/synthetic/e0_design_input_lock.pt",
            "bytes": 1,
            "file_sha256": sha,
            "semantic_sha256": sha,
            "item_count": 326,
            "choice_audit_lock_sha256": choice_binding["lock"]["sha256"],
            "expanded_logical_rows_present": False,
            "custom_verifier": {
                "exact_top_level_key_set_verified": True,
                "exact_item_key_set_verified": True,
                "semantic_sha256_recomputed_after_cpu_reload": True,
                "choice_audit_lock_sha256_recomputed_and_equal": True,
                "expanded_logical_rows_present": False,
                "model_or_checkpoint_tensor_present": False,
            },
        },
        "streaming": {
            "protocol_id": "fpct_e1_mechanism_audit_v6_representation_preserving_streaming",
            "schema_sha256": sha,
            "physical_chunk_rows": 4096,
            "synthetic_gate_sha256": sha,
            "geometry_lock_receipt_sha256": sha,
            "streaming_template_lock_receipt_sha256": sha,
            "logical_row_coverage_exact": True,
            "streaming_semantic_replay_equal": True,
            "whole_table_materialization_detected": False,
        },
        "hard_gate_checks": hard_checks,
        "hash_domain_checks": hash_checks,
        "zero_counts": {
            "unexpected_prompt_difference_count": 0,
            "missing_rows": 0,
            "duplicate_rows": 0,
        },
        "firewall": firewall,
        "e1_2_or_e1_3_authorized": False,
    }


def test_a5r3_publication_is_independently_verified_but_v9_manifest_binding_is_exact_five_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, _dev, _examples = _run_synthetic_audit(tmp_path, monkeypatch)
    assert "publication" in verified
    _strict_tmp_aware_a5r3_validator(verified["publication"], _identity(tmp_path))
    binding = prepare._choice_audit_binding(
        verified, _identity(tmp_path / "synthetic-a5r2-run")
    )
    assert set(binding) == {
        "lock",
        "ledger",
        "status",
        "audit_semantic_sha256",
        "correction_actions_selected",
    }
    assert "publication" not in binding
    manifest = _complete_v9_input_manifest(binding)
    a5r2_gate.validate_a5r2_schema_artifact(manifest, repo_root=REPO_ROOT)
    contaminated = copy.deepcopy(manifest)
    contaminated["choice_audit"]["publication"] = verified["publication"]
    with pytest.raises(ValueError, match="oneOf matched 0 branches"):
        a5r2_gate.validate_a5r2_schema_artifact(contaminated, repo_root=REPO_ROOT)


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
    with pytest.raises(
        RuntimeError,
        match="publication receipt binding|frozen membership/order",
    ):
        prepare._verify_a5r2_choice_audit(
            repo_root=REPO_ROOT,
            e0_data_root=tmp_path / "synthetic-e0-data",
            execution_identity=_identity(run_root),
            a5_contract={"synthetic": True},
            require_go=False,
        )


def _stub_a5r7_current_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from script.analysis import fpct_e1_a5r7_gate

    monkeypatch.setattr(
        fpct_e1_a5r7_gate,
        "verify_gate",
        lambda *_args, **_kwargs: {
            "checks": {
                "historical_a5r4_gate_consumed_without_live_tree_replay": True,
                "successor_natural_accessed": False,
            }
        },
    )


def test_a5r7_production_preflight_uses_current_gate_and_immutable_a5r4_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from script.analysis import fpct_e1_a5r4_setgid_mode_gate as a5r4_gate
    from script.analysis import fpct_e1_a5r7_gate

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(
        fpct_e1_a5r7_gate,
        "verify_gate",
        lambda _path, repo_root: fpct_e1_a5r7_gate.verify_pre_output_candidate(
            repo_root
        ),
    )
    active = prepare._load_active_a5_synthetic_gate(REPO_ROOT)
    assert active["schema_version"] == 11
    assert active["choice_semantics_version"] == 9
    assert active["status"] == prepare.A5_SYNTHETIC_GATE_STATUS
    assert active["inherited_v8_streaming_evidence"]
    source = inspect.getsource(prepare._load_active_a5_synthetic_gate)
    assert "verify_current_gate" in source
    assert "verify_a5r4_gate" not in source
    assert prepare.A5R4_SYNTHETIC_GATE_SHA256 == prepare.sha256_file(
        REPO_ROOT / prepare.A5R4_HISTORICAL_GATE_RELATIVE
    )
    assert a5r4_gate.verify_mode_predicate_static(REPO_ROOT)


def test_a5r7_historical_live_verifier_is_never_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from script.analysis import fpct_e1_a5r4_setgid_mode_gate as a5r4_gate

    _stub_a5r7_current_gate(monkeypatch)
    monkeypatch.setattr(
        a5r4_gate,
        "verify_gate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("historical live-tree verifier must not run")
        ),
    )
    assert prepare._load_active_a5_synthetic_gate(REPO_ROOT)["schema_version"] == 11


def test_a5r7_historical_gate_tamper_precedes_schema_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from script.analysis import fpct_e1_a5r4_setgid_mode_gate as a5r4_gate

    _stub_a5r7_current_gate(monkeypatch)
    historical = REPO_ROOT / prepare.A5R4_HISTORICAL_GATE_RELATIVE
    tampered = tmp_path / "tampered-a5r4.json"
    tampered.write_bytes(historical.read_bytes() + b" ")
    monkeypatch.setattr(prepare, "A5R4_HISTORICAL_GATE_RELATIVE", tampered)
    monkeypatch.setattr(prepare, "_load_a5r4_setgid_mode_contract", lambda _root: {})
    called = False

    def forbidden_validator(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("schema validator must follow exact byte check")

    monkeypatch.setattr(a5r4_gate, "validate_a5r4_schema_artifact", forbidden_validator)
    with pytest.raises(ValueError, match="immutable A5R4 synthetic gate changed"):
        prepare._load_active_a5_synthetic_gate(REPO_ROOT)
    assert called is False


def test_a5r7_historical_gate_alias_and_nonregular_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_a5r7_current_gate(monkeypatch)
    historical = REPO_ROOT / prepare.A5R4_HISTORICAL_GATE_RELATIVE
    alias = tmp_path / "alias.json"
    alias.symlink_to(historical)
    monkeypatch.setattr(prepare, "A5R4_HISTORICAL_GATE_RELATIVE", alias)
    with pytest.raises(RuntimeError, match="non-symlink regular file"):
        prepare._load_active_a5_synthetic_gate(REPO_ROOT)
    directory = tmp_path / "directory"
    directory.mkdir()
    monkeypatch.setattr(prepare, "A5R4_HISTORICAL_GATE_RELATIVE", directory)
    with pytest.raises(RuntimeError, match="non-symlink regular file"):
        prepare._load_active_a5_synthetic_gate(REPO_ROOT)


def test_a5r7_strict_schema_and_identity_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from script.analysis import fpct_e1_a5r4_setgid_mode_gate as a5r4_gate

    _stub_a5r7_current_gate(monkeypatch)
    historical = json.loads(
        (REPO_ROOT / prepare.A5R4_HISTORICAL_GATE_RELATIVE).read_text(encoding="utf-8")
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_bytes(prepare.canonical_json_bytes(historical))
    monkeypatch.setattr(prepare, "A5R4_HISTORICAL_GATE_RELATIVE", candidate)
    monkeypatch.setattr(prepare, "_load_a5r4_setgid_mode_contract", lambda _root: {})
    monkeypatch.setattr(
        prepare, "A5R4_SYNTHETIC_GATE_SHA256", prepare.sha256_file(candidate)
    )
    monkeypatch.setattr(
        a5r4_gate,
        "validate_a5r4_schema_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("strict-v11")),
    )
    with pytest.raises(ValueError, match="strict-v11"):
        prepare._load_active_a5_synthetic_gate(REPO_ROOT)

    wrong = dict(historical)
    wrong["status"] = "WRONG"
    candidate.write_bytes(prepare.canonical_json_bytes(wrong))
    monkeypatch.setattr(
        prepare, "A5R4_SYNTHETIC_GATE_SHA256", prepare.sha256_file(candidate)
    )
    monkeypatch.setattr(a5r4_gate, "validate_a5r4_schema_artifact", lambda *_a, **_k: None)
    monkeypatch.setattr(
        a5r4_gate, "_evidence_sha256", lambda value: value["evidence_sha256"]
    )
    with pytest.raises(ValueError, match="identity changed"):
        prepare._load_active_a5_synthetic_gate(REPO_ROOT)


def test_a5r7_creation_and_completed_replay_share_preflight() -> None:
    creation = inspect.getsource(prepare._prepare_input_lock_after_identity)
    completed = inspect.getsource(prepare._verify_completed_input_lock)
    assert creation.count("_load_active_a5_synthetic_gate(") == 1
    assert completed.count("_load_active_a5_synthetic_gate(") == 1
