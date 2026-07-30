from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from script.analysis.fpct_e1_streaming_verify import _validate_json_schema
from script.evaluation.unified_evaluator import UnifiedEvaluator
from script.experiment.fpct_e1_a5_prompt_provenance import (
    A5_PROTOCOL_ID,
    A5R1_PROTOCOL_ID,
    A5R2_AMENDMENT_ID,
    A5R2_PROTOCOL_ID,
    EXTRA_CHOICES_ONLY,
    HISTORICAL_EXACT,
    build_census_manifest,
    build_choice_cardinality_audit_lock_v9,
    canonical_historical_content_sha256,
    choice_audit_projection_v9,
    choice_cardinality_record_v9,
    choice_payload_v9,
    dual_anchor_record,
    full_question_choices_v9,
    historical_projected_example_v9,
    sha256_bytes,
    summarize_choice_cardinality_audit_v9,
    summarize_dual_anchor_census,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = (
    REPO_ROOT
    / "recipe/eval_recipe/fpct_e1/e1_a5r2_choice_cardinality_schema.json"
)
EXECUTION_SHA = "a" * 40
RUN_UID = "fpct-e1-a5r2-choice-cardinality-aaaaaaaa-v1"
RUN_ROOT = "/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-aaaaaaaa-v1"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validate(value: dict[str, Any]) -> None:
    schema = _schema()
    _validate_json_schema(value, schema, schema, path="$")


def _arc(count: int, *, numeric_labels: bool = False) -> dict[str, Any]:
    labels = (
        [str(index + 1) for index in range(count)]
        if numeric_labels
        else [chr(65 + index) for index in range(count)]
    )
    return {
        "question": "Which answer?",
        "choices": {
            "text": [f"choice-{index}" for index in range(count)],
            "label": labels,
        },
        "answerKey": "A",
        "correctness": "must-not-read",
    }


def _obqa(count: int = 4) -> dict[str, Any]:
    return {
        "question_stem": "Which fact?",
        "choices": {
            "text": [f"fact-{index}" for index in range(count)],
            "label": [chr(65 + index) for index in range(count)],
        },
        "answerKey": "A",
    }


def _mmlu(count: int = 4) -> dict[str, Any]:
    return {
        "question": "Which theorem?",
        "choices": [f"theorem-{index}" for index in range(count)],
        "answer": 0,
    }


def _descriptor(
    *,
    task: str,
    index: int,
    mismatch: bool = False,
) -> dict[str, Any]:
    dataset = {
        "ai2-arc": ("allenai/ai2_arc", "ARC-Challenge"),
        "openbookqa": ("openbookqa", "main"),
        "mmlu-redux": ("edinburgh-dawg/mmlu-redux-2.0", "subject"),
    }[task]
    expected = {
        "dataset_id": dataset[0],
        "dataset_config": dataset[1],
        "dataset_split": "test",
        "evaluation_question_id": index,
        "non_outcome_native_row_id": str(index),
        "feature_schema_sha256": _sha(f"feature-{task}"),
    }
    observed = dict(expected)
    if mismatch:
        observed["evaluation_question_id"] = index + 1
    return {"expected": expected, "observed": observed}


def _formatter(task: str, example: dict[str, Any]) -> str:
    formatter = UnifiedEvaluator.__new__(UnifiedEvaluator)
    formatter.dataset_name = task
    formatter.model_config = {}
    formatter.eval_config = {
        "dataset": task,
        "use_cot": False,
        "use_template": True,
    }
    return formatter.format_example(example, use_cot=False)


def _record(
    *,
    ordinal: int,
    task: str,
    example: dict[str, Any],
    mismatch: bool = False,
    legacy_choice_count: int | None = None,
) -> dict[str, Any]:
    descriptor = _descriptor(task=task, index=ordinal - 1, mismatch=mismatch)
    projection = choice_audit_projection_v9(task, descriptor, example)
    raw_prompt = _formatter(task, projection["raw_formatter_example"])
    canonical_prompt = _formatter(task, projection["canonical_formatter_example"])
    prompt_equal = raw_prompt == canonical_prompt
    count = int(projection["persisted"]["production_choice_count"])
    group = projection["persisted"]["observed_content_group_sha256"]
    return choice_cardinality_record_v9(
        execution_sha=EXECUTION_SHA,
        run_uid=RUN_UID,
        group_ordinal=ordinal,
        task=task,
        content_group_sha256=("f" * 64 if mismatch else group),
        sample_key_sha256=_sha(f"sample-{ordinal}"),
        observed_sample_key_sha256=_sha(f"sample-{ordinal}"),
        descriptor=descriptor,
        projection=projection,
        raw_prompt_sha256=sha256_bytes(raw_prompt.encode("utf-8")),
        canonical_prompt_sha256=sha256_bytes(canonical_prompt.encode("utf-8")),
        formatter_byte_parity=prompt_equal,
        legacy_choice_count=(count if legacy_choice_count is None else legacy_choice_count),
    )


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_arc_cardinality_preserves_full_production_and_min_four_history(
    count: int,
) -> None:
    example = _arc(count, numeric_labels=True)
    original = copy.deepcopy(example)
    question, choices, source_labels, runtime_labels = full_question_choices_v9(
        "ai2-arc", example
    )
    assert question == example["question"]
    assert len(choices) == count
    assert source_labels == [str(index + 1) for index in range(count)]
    assert runtime_labels == [chr(65 + index) for index in range(count)]
    projected = historical_projected_example_v9("ai2-arc", example)
    assert len(projected["choices"]["text"]) == min(4, count)
    assert example == original


@pytest.mark.parametrize(
    ("task", "example", "message"),
    [
        ("ai2-arc", _arc(1), "fewer than two"),
        ("openbookqa", _obqa(3), "not exactly four"),
        ("openbookqa", _obqa(5), "not exactly four"),
        ("mmlu-redux", _mmlu(3), "not exactly four"),
        ("mmlu-redux", _mmlu(5), "not exactly four"),
    ],
)
def test_production_parser_enforces_task_cardinality(
    task: str, example: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        choice_payload_v9(task, example)
    # The audit view must classify the same row instead of failing before the
    # prospective task-cardinality taxonomy can run.
    projection = choice_audit_projection_v9(task, _descriptor(task=task, index=0), example)
    assert projection["persisted"]["production_choice_count"] == len(
        example["choices"]["text"]
        if isinstance(example["choices"], dict)
        else example["choices"]
    )


def test_mapping_and_sequence_of_mappings_are_distinct_supported_views() -> None:
    mapping = _arc(3)
    sequence = {
        "question": mapping["question"],
        "choices": [
            {"text": text, "label": label}
            for text, label in zip(
                mapping["choices"]["text"], mapping["choices"]["label"]
            )
        ],
    }
    mapped = choice_payload_v9("ai2-arc", mapping)
    sequenced = choice_payload_v9("ai2-arc", sequence)
    assert mapped["choice_container_kind"] == "mapping_of_sequences"
    assert sequenced["choice_container_kind"] == "sequence_of_mappings"
    assert mapped["production_choices"] == sequenced["production_choices"]
    assert mapped["source_choice_labels"] == sequenced["source_choice_labels"]


def test_direct_tuple_that_runtime_formatter_cannot_preserve_is_a_hard_stop() -> None:
    """Canonical support alone cannot turn a runtime formatting mismatch into GO."""

    base = _arc(4)
    example = {
        "question": base["question"],
        "choices": tuple(
            {"text": text, "label": label}
            for text, label in zip(
                base["choices"]["text"], base["choices"]["label"]
            )
        ),
    }
    record = _record(ordinal=1, task="ai2-arc", example=example)
    assert record["representation_status"] == "DIRECT"
    assert record["formatter_byte_parity"] is False
    assert record["root_cause_class"] == "UNEXPLAINED_OTHER"
    assert record["correction_action"] == "BLOCK"
    _validate(record)


class _FeatureSequence:
    def __init__(self, values: list[Any]):
        self.values = values

    def tolist(self) -> list[Any]:
        return list(self.values)

    def __iter__(self):
        return iter(self.values)


class _ToPylistFeatureSequence:
    def __init__(self, values: list[Any]):
        self.values = values

    def to_pylist(self) -> list[Any]:
        return list(self.values)

    def __iter__(self):
        return iter(self.values)


class _AsPyFeatureSequence:
    def __init__(self, values: list[Any]):
        self.values = values

    def as_py(self) -> list[Any]:
        return list(self.values)

    def __iter__(self):
        return iter(self.values)


class _AmbiguousFeatureSequence(_FeatureSequence):
    def as_py(self) -> list[Any]:
        return list(self.values)


class _GeneratorFeatureSequence:
    def tolist(self):
        return (value for value in range(4))


def test_feature_backed_conversion_is_unique_finite_and_formatter_equal() -> None:
    example = _arc(4)
    example["choices"] = {
        "text": _FeatureSequence(example["choices"]["text"]),
        "label": _FeatureSequence(example["choices"]["label"]),
    }
    record = _record(
        ordinal=1,
        task="ai2-arc",
        example=example,
        legacy_choice_count=0,
    )
    assert record["materialized_container_kind"] == "feature_backed_sequence_to_list"
    assert record["representation_status"] == "HF_FEATURE_NORMALIZATION_REQUIRED"
    assert record["root_cause_class"] == "HF_FEATURE_NORMALIZATION_ONLY"
    assert record["formatter_byte_parity"] is True
    _validate(record)


@pytest.mark.parametrize(
    "wrapper",
    (_FeatureSequence, _ToPylistFeatureSequence, _AsPyFeatureSequence),
)
def test_feature_conversion_survives_audit_then_historical_projection(
    wrapper: type,
) -> None:
    example = _arc(5)
    example["choices"] = {
        "text": wrapper(example["choices"]["text"]),
        "label": wrapper(example["choices"]["label"]),
    }
    projection = choice_audit_projection_v9(
        "ai2-arc", _descriptor(task="ai2-arc", index=0), example
    )
    assert projection["persisted"]["production_choice_count"] == 5
    assert projection["persisted"]["representation_status"] == (
        "HF_FEATURE_NORMALIZATION_REQUIRED"
    )
    projected = historical_projected_example_v9("ai2-arc", example)
    assert projected["choices"]["text"] == [
        "choice-0",
        "choice-1",
        "choice-2",
        "choice-3",
    ]
    assert len(choice_payload_v9("ai2-arc", example)["production_choices"]) == 5


def test_task_invalid_and_descriptor_mismatch_receive_predeclared_block_classes() -> None:
    invalid = _record(ordinal=1, task="openbookqa", example=_obqa(3))
    assert invalid["cardinality_status"] == "TASK_INVALID"
    assert invalid["root_cause_class"] == "GENUINE_TASK_INVALID_CARDINALITY"
    assert invalid["correction_action"] == "BLOCK"
    _validate(invalid)

    mismatch = _record(
        ordinal=1,
        task="ai2-arc",
        example=_arc(3),
        mismatch=True,
    )
    assert mismatch["identity_status"] == "DESCRIPTOR_INDEX_MISMATCH"
    assert mismatch["root_cause_class"] == "DESCRIPTOR_INDEX_MISMATCH"
    assert mismatch["correction_action"] == "BLOCK"
    _validate(mismatch)


@pytest.mark.parametrize(
    ("count", "root_cause", "action"),
    [
        (2, "GENUINE_ARC_LOW_CARDINALITY", "VARIABLE_CARDINALITY_CONTRACT_ONLY"),
        (3, "GENUINE_ARC_LOW_CARDINALITY", "VARIABLE_CARDINALITY_CONTRACT_ONLY"),
        (4, "DIRECT_VALID", "NONE"),
        (5, "DIRECT_VALID", "NONE"),
    ],
)
def test_arc_root_cause_and_action_are_mechanical(
    count: int, root_cause: str, action: str
) -> None:
    record = _record(ordinal=1, task="ai2-arc", example=_arc(count))
    assert record["root_cause_class"] == root_cause
    assert record["correction_action"] == action
    _validate(record)


@pytest.mark.parametrize(
    "choices",
    [
        "ABCD",
        (value for value in range(4)),
        [{"text": "a", "label": "A"}, "ambiguous"],
        {"text": ["a", "b"], "label": ["A"]},
        {"text": ["a", "b"], "label": ["A", "A"]},
        _AmbiguousFeatureSequence([{"text": "a", "label": "A"}]),
        _GeneratorFeatureSequence(),
    ],
)
def test_ambiguous_or_unbounded_choice_representations_fail_closed(choices: Any) -> None:
    with pytest.raises(ValueError):
        choice_payload_v9(
            "ai2-arc",
            {"question": "q", "choices": choices},
            enforce_task_cardinality=False,
        )


@pytest.mark.parametrize(
    ("task", "example"),
    [
        ("ai2-arc", {**_arc(4), "question": 123}),
        ("openbookqa", {**_obqa(4), "question_stem": ["not", "text"]}),
        ("mmlu-redux", {**_mmlu(4), "question": {"not": "text"}}),
    ],
)
def test_question_representation_is_never_silently_stringified(
    task: str, example: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="nonempty string"):
        choice_payload_v9(task, example, enforce_task_cardinality=False)


class _OutcomeFirewallRow(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed: list[str] = []

    def get(self, key, default=None):
        self.accessed.append(str(key))
        if key in {"answer", "answerKey", "correct_answer", "correctness", "logits"}:
            raise AssertionError(f"forbidden field accessed: {key}")
        return super().get(key, default)


def test_choice_projection_never_reads_or_hashes_outcome_fields() -> None:
    first = _OutcomeFirewallRow(_arc(3))
    first["answerKey"] = "A"
    first["correctness"] = object()
    second = _OutcomeFirewallRow(_arc(3))
    second["answerKey"] = "D"
    second["correctness"] = object()
    descriptor = _descriptor(task="ai2-arc", index=0)
    first_projection = choice_audit_projection_v9("ai2-arc", descriptor, first)
    second_projection = choice_audit_projection_v9("ai2-arc", descriptor, second)
    assert first_projection["persisted"] == second_projection["persisted"]
    assert set(first.accessed).isdisjoint(
        {"answer", "answerKey", "correct_answer", "correctness", "logits"}
    )
    assert "answerKey" not in first_projection["raw_formatter_example"]
    assert "correctness" not in first_projection["raw_formatter_example"]


def test_numeric_source_labels_are_not_gold_authority() -> None:
    example = _arc(3, numeric_labels=True)
    projected = historical_projected_example_v9("ai2-arc", example)
    historical_prompt = _formatter("ai2-arc", projected)
    production_prompt = _formatter("ai2-arc", example)
    group = canonical_historical_content_sha256(
        example["question"], example["choices"]["text"]
    )
    record = dual_anchor_record(
        task="ai2-arc",
        content_group_sha256=group,
        sample_key_sha256="2" * 64,
        source_row_id="1",
        example=example,
        gold_answer="A",
        historical_rendered_prompt=historical_prompt,
        historical_alignment_sha256="3" * 64,
        production_rendered_prompt=production_prompt,
        production_alignment_sha256="3" * 64,
        historical_prompt_token_count=10,
        production_prompt_token_count=10,
        production_certified_parent_count=2,
        production_logical_row_count=3,
        production_physical_chunk_count=1,
        historical_content_sha256=group,
        schema_version=9,
        protocol_id=A5R2_PROTOCOL_ID,
    )
    assert record["source_choice_labels"] == ["1", "2", "3"]
    assert record["runtime_ordinal_labels"] == ["A", "B", "C"]
    assert record["gold_answer"] == "A"
    assert record["prompt_relation"] == HISTORICAL_EXACT


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_v9_dual_anchor_relation_is_exact_or_suffix_only(count: int) -> None:
    example = _arc(count)
    projected = historical_projected_example_v9("ai2-arc", example)
    historical_prompt = _formatter("ai2-arc", projected)
    production_prompt = _formatter("ai2-arc", example)
    group = canonical_historical_content_sha256(
        example["question"], example["choices"]["text"]
    )
    record = dual_anchor_record(
        task="ai2-arc",
        content_group_sha256=group,
        sample_key_sha256="2" * 64,
        source_row_id="1",
        example=example,
        gold_answer="A",
        historical_rendered_prompt=historical_prompt,
        historical_alignment_sha256="3" * 64,
        production_rendered_prompt=production_prompt,
        production_alignment_sha256=("3" * 64 if count <= 4 else "4" * 64),
        historical_prompt_token_count=10,
        production_prompt_token_count=10 + max(0, count - 4),
        production_certified_parent_count=2,
        production_logical_row_count=3,
        production_physical_chunk_count=1,
        historical_content_sha256=group,
        schema_version=9,
        protocol_id=A5R2_PROTOCOL_ID,
    )
    assert record["historical_choice_count"] == min(4, count)
    assert record["production_choice_count"] == count
    assert record["prompt_relation"] == (
        HISTORICAL_EXACT if count <= 4 else EXTRA_CHOICES_ONLY
    )


def test_permutation_is_preserved_not_silently_canonicalized() -> None:
    first = _arc(4)
    second = copy.deepcopy(first)
    order = [2, 0, 3, 1]
    second["choices"]["text"] = [first["choices"]["text"][index] for index in order]
    second["choices"]["label"] = [first["choices"]["label"][index] for index in order]
    descriptor = _descriptor(task="ai2-arc", index=0)
    first_projection = choice_audit_projection_v9("ai2-arc", descriptor, first)
    second_projection = choice_audit_projection_v9("ai2-arc", descriptor, second)
    assert (
        first_projection["persisted"]["choice_projection_sha256"]
        != second_projection["persisted"]["choice_projection_sha256"]
    )
    assert _formatter("ai2-arc", first_projection["raw_formatter_example"]) != _formatter(
        "ai2-arc", second_projection["raw_formatter_example"]
    )


def test_v7_v8_default_path_remains_semantically_unchanged() -> None:
    example = _arc(4)
    group = canonical_historical_content_sha256(
        example["question"], example["choices"]["text"]
    )
    kwargs = dict(
        task="ai2-arc",
        content_group_sha256=group,
        sample_key_sha256="2" * 64,
        source_row_id="1",
        example=example,
        gold_answer="A",
        historical_rendered_prompt="same",
        historical_alignment_sha256="3" * 64,
        production_rendered_prompt="same",
        production_alignment_sha256="3" * 64,
        historical_prompt_token_count=1,
        production_prompt_token_count=1,
        production_certified_parent_count=1,
        production_logical_row_count=1,
        production_physical_chunk_count=1,
        historical_content_sha256=group,
    )
    default = dual_anchor_record(**kwargs)
    explicit_v7 = dual_anchor_record(
        **kwargs, schema_version=7, protocol_id=A5_PROTOCOL_ID
    )
    v8 = dual_anchor_record(
        **kwargs, schema_version=8, protocol_id=A5R1_PROTOCOL_ID
    )
    assert default == explicit_v7
    assert {key: value for key, value in v8.items() if key not in {"schema_version", "protocol_id"}} == {
        key: value for key, value in default.items() if key not in {"schema_version", "protocol_id"}
    }
    assert A5R2_AMENDMENT_ID.endswith("_RECOVERY")


def _full_population(*, blocked: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordinal = 1
    for task, count in (("ai2-arc", 128), ("openbookqa", 70), ("mmlu-redux", 128)):
        for task_index in range(count):
            if task == "ai2-arc":
                example = _arc((2, 3, 4, 5)[task_index % 4])
            elif task == "openbookqa":
                example = _obqa()
            else:
                example = _mmlu()
            question_field = "question_stem" if task == "openbookqa" else "question"
            example[question_field] = f"{example[question_field]} row-{ordinal}"
            rows.append(
                _record(
                    ordinal=ordinal,
                    task=task,
                    example=example,
                    mismatch=blocked and ordinal == 1,
                )
            )
            ordinal += 1
    return rows


def _audit_lock_inputs(summary: dict[str, Any]) -> dict[str, Any]:
    summary_payload = json.dumps(
        summary,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return {
        "ledger_artifact": {
            "relative_path": "choice_audit/rows.jsonl",
            "sha256": "b" * 64,
            "bytes": 1234,
            "row_count": 326,
        },
        "summary_artifact": {
            "relative_path": "choice_audit/summary.json",
            "sha256": hashlib.sha256(summary_payload).hexdigest(),
            "bytes": len(summary_payload),
            "row_count": 1,
        },
        "e0_data_tree_binding": {
            "file_count": 51,
            "bytes": 233569,
            "e0_declared_tree_algorithm": "relative_path_nul_file_sha256_bytes_v1",
            "e0_declared_tree_sha256": "f3dcf2c77e6c5f90946994488fcb86f67dcdc590510a9f469f32e86773492c73",
            "generic_asset_tree_algorithm": "canonical_json_file_manifest_v1",
            "generic_asset_tree_sha256": "12f537cade1a30f6fd4e7a146c58311412f8824a6845ea4e6f7a6b5651bcb405",
            "both_domains_recomputed_before_audit_row_one": True,
            "cross_domain_comparison_detected": False,
        },
        "source_snapshot_binding": {
            "receipt_path": "source_snapshot/.receipt.json",
            "receipt_file_sha256": "d" * 64,
            "receipt_sha256": "e" * 64,
            "mounted_tree_sha256": "f" * 64,
            "execution_sha": EXECUTION_SHA,
            "independently_verified_before_audit_row_one": True,
        },
        "source_bindings": [{"path": "repo://manifest", "sha256": "c" * 64}],
        "predecessor_v8_unchanged": True,
        "independent_reduction_and_hash_verified": True,
        "frozen_group_sample_membership_verified": True,
        "frozen_canonical_order_verified": True,
        "frozen_loader_coordinates_verified": True,
    }


@pytest.mark.parametrize("blocked", [False, True])
def test_full_population_go_and_blocked_locks_validate_strict_schema(
    blocked: bool,
) -> None:
    rows = _full_population(blocked=blocked)
    for row in rows:
        _validate(row)
    summary = summarize_choice_cardinality_audit_v9(rows)
    expected_status = (
        "A5R2_CHOICE_AUDIT_BLOCKED" if blocked else "A5R2_CHOICE_AUDIT_GO"
    )
    assert summary["status"] == expected_status
    lock = build_choice_cardinality_audit_lock_v9(
        rows,
        execution_sha=EXECUTION_SHA,
        run_uid=RUN_UID,
        run_root=RUN_ROOT,
        **_audit_lock_inputs(summary),
    )
    assert lock["status"] == expected_status
    assert lock["same_execution_input_lock_consumption_allowed"] is (not blocked)
    assert lock["hard_checks"]["e0_data_tree_dual_domains_verified_before_audit"]
    assert lock["hard_checks"][
        "source_snapshot_receipt_and_tree_verified_before_audit"
    ]
    _validate(lock)


@pytest.mark.parametrize(
    ("task", "population_index", "count"),
    [
        ("openbookqa", 128, 3),
        ("mmlu-redux", 198, 5),
    ],
)
def test_complete_population_with_parseable_invalid_cardinality_mechanically_blocks(
    task: str,
    population_index: int,
    count: int,
) -> None:
    """A parseable invalid row remains in the complete 326-row BLOCK ledger."""

    rows = _full_population()
    ordinal = population_index + 1
    example = _obqa(count) if task == "openbookqa" else _mmlu(count)
    question_field = "question_stem" if task == "openbookqa" else "question"
    example[question_field] = f"{example[question_field]} invalid-row-{ordinal}"
    rows[population_index] = _record(
        ordinal=ordinal,
        task=task,
        example=example,
    )

    invalid = rows[population_index]
    assert invalid["canonical_choice_count"] == count
    assert invalid["source_choice_label_count"] == count
    assert invalid["runtime_ordinal_label_count"] == count
    assert invalid["root_cause_class"] == "GENUINE_TASK_INVALID_CARDINALITY"
    assert invalid["correction_action"] == "BLOCK"
    for row in rows:
        _validate(row)

    summary = summarize_choice_cardinality_audit_v9(rows)
    assert summary["row_count"] == 326
    assert summary["status"] == "A5R2_CHOICE_AUDIT_BLOCKED"
    assert summary["root_cause_counts"]["GENUINE_TASK_INVALID_CARDINALITY"] == 1
    lock = build_choice_cardinality_audit_lock_v9(
        rows,
        execution_sha=EXECUTION_SHA,
        run_uid=RUN_UID,
        run_root=RUN_ROOT,
        **_audit_lock_inputs(summary),
    )
    assert lock["status"] == "A5R2_CHOICE_AUDIT_BLOCKED"
    assert lock["same_execution_input_lock_consumption_allowed"] is False
    _validate(lock)


def test_audit_lock_rejects_unbound_summary_dual_domain_or_snapshot() -> None:
    rows = _full_population()
    summary = summarize_choice_cardinality_audit_v9(rows)
    valid = _audit_lock_inputs(summary)

    bad_summary = copy.deepcopy(valid)
    bad_summary["summary_artifact"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="summary descriptor"):
        build_choice_cardinality_audit_lock_v9(
            rows,
            execution_sha=EXECUTION_SHA,
            run_uid=RUN_UID,
            run_root=RUN_ROOT,
            **bad_summary,
        )

    bad_domains = copy.deepcopy(valid)
    bad_domains["e0_data_tree_binding"]["cross_domain_comparison_detected"] = True
    with pytest.raises(ValueError, match="dual-domain"):
        build_choice_cardinality_audit_lock_v9(
            rows,
            execution_sha=EXECUTION_SHA,
            run_uid=RUN_UID,
            run_root=RUN_ROOT,
            **bad_domains,
        )

    bad_snapshot = copy.deepcopy(valid)
    bad_snapshot["source_snapshot_binding"]["execution_sha"] = "0" * 40
    with pytest.raises(ValueError, match="source snapshot"):
        build_choice_cardinality_audit_lock_v9(
            rows,
            execution_sha=EXECUTION_SHA,
            run_uid=RUN_UID,
            run_root=RUN_ROOT,
            **bad_snapshot,
        )


def _v9_census_population(choice_audit_lock_sha256: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    ordinal = 1
    for task, task_count in (
        ("ai2-arc", 128),
        ("openbookqa", 70),
        ("mmlu-redux", 128),
    ):
        for task_index in range(task_count):
            if task == "ai2-arc":
                count = (2, 3, 4, 5)[task_index % 4]
                example = _arc(count)
            elif task == "openbookqa":
                count, example = 4, _obqa()
            else:
                count, example = 4, _mmlu()
            question_field = "question_stem" if task == "openbookqa" else "question"
            example[question_field] = f"{example[question_field]} census-{ordinal}"
            projected = historical_projected_example_v9(task, example)
            historical_prompt = _formatter(task, projected)
            production_prompt = _formatter(task, example)
            payload = choice_payload_v9(task, example)
            group = canonical_historical_content_sha256(
                payload["question"], payload["production_choices"]
            )
            record = dual_anchor_record(
                task=task,
                content_group_sha256=group,
                sample_key_sha256=_sha(f"census-sample-{ordinal}"),
                source_row_id=str(ordinal - 1),
                example=example,
                gold_answer="A",
                historical_rendered_prompt=historical_prompt,
                historical_alignment_sha256="1" * 64,
                production_rendered_prompt=production_prompt,
                production_alignment_sha256=(
                    "1" * 64 if count <= 4 else "2" * 64
                ),
                historical_prompt_token_count=10,
                production_prompt_token_count=10 + max(0, count - 4),
                production_certified_parent_count=2,
                production_logical_row_count=3,
                production_physical_chunk_count=1,
                historical_content_sha256=group,
                schema_version=9,
                protocol_id=A5R2_PROTOCOL_ID,
            )
            record["choice_audit_row_sha256"] = _sha(f"audit-row-{ordinal}")
            record["choice_audit_lock_sha256"] = choice_audit_lock_sha256
            record["choice_root_cause_class"] = (
                "GENUINE_ARC_LOW_CARDINALITY"
                if task == "ai2-arc" and count in {2, 3}
                else "DIRECT_VALID"
            )
            records.append(record)
            ordinal += 1
    task_rank = {task: index for index, task in enumerate(("ai2-arc", "openbookqa", "mmlu-redux"))}
    return sorted(
        records,
        key=lambda row: (
            task_rank[row["task"]],
            row["content_group_sha256"],
            row["sample_key_sha256"],
        ),
    )


def test_v9_census_records_summary_and_manifest_validate_strict_schema() -> None:
    choice_audit_lock_sha256 = "9" * 64
    records = _v9_census_population(choice_audit_lock_sha256)
    for record in records:
        _validate(record)
    summary = summarize_dual_anchor_census(
        records,
        schema_version=9,
        protocol_id=A5R2_PROTOCOL_ID,
        choice_audit_lock_sha256=choice_audit_lock_sha256,
    )
    assert summary["choice_audit_lock_sha256"] == choice_audit_lock_sha256
    _validate(summary)
    manifest = build_census_manifest(
        records,
        execution_sha=EXECUTION_SHA,
        run_uid=RUN_UID,
        record_artifact={
            "relative_path": "input_lock/a5_prompt_census_records.jsonl",
            "sha256": "8" * 64,
            "bytes": 12345,
            "row_count": 326,
        },
        schema_version=9,
        protocol_id=A5R2_PROTOCOL_ID,
        choice_audit_lock_sha256=choice_audit_lock_sha256,
    )
    assert manifest["choice_audit_lock_sha256"] == choice_audit_lock_sha256
    assert all(
        "choice_audit_row_sha256" in row
        and "choice_root_cause_class" in row
        and "label_free_runtime_row_sha256" in row
        for row in manifest["historical_to_production_anchor_map"]
    )
    _validate(manifest)
