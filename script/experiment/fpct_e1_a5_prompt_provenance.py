#!/usr/bin/env python3
"""Pure, no-model dual-anchor provenance primitives for FPCT-E1 A5.

The historical first-four projection and the actual E0 production runtime
prompt are deliberately different objects.  This module makes the distinction
executable without importing a tokenizer, model class, checkpoint, CUDA or
Kubernetes client.  Natural-data access remains the responsibility of the
sealed A5 input-lock producer.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


A5_PROTOCOL_ID = "fpct_e1_mechanism_audit_v7_actual_e0_runtime_prompt"
A5_AMENDMENT_ID = "APPROVED_PROSPECTIVE_AMENDMENT_E1_A5_RUNTIME_PROMPT"
HISTORICAL_EXACT = "EXACT_HISTORICAL_AND_PRODUCTION_MATCH"
EXTRA_CHOICES_ONLY = "EXTRA_CHOICES_ONLY"
PROMPT_RELATIONS = frozenset((HISTORICAL_EXACT, EXTRA_CHOICES_ONLY))
TASK_ORDER = ("ai2-arc", "openbookqa", "mmlu-redux")
EXPECTED_TASK_COUNTS = {"ai2-arc": 128, "openbookqa": 70, "mmlu-redux": 128}
GOLD_ANSWERS = frozenset(("A", "B", "C", "D"))

# 80fb295 is the original scientific source.  The effective decode-recovery
# evaluation image is 6a51ad4.  Its complete evaluator/aligner/data-loading
# closure is frozen below; unrelated later helpers were added to evaluate.py,
# so its two prompt-relevant functions are bound by exact AST source segments.
E0_SCIENTIFIC_CODE_COMMIT = "80fb295542ad298fae4cddb1273517b401bbcd17"
E0_FINAL_EVALUATION_SOURCE_COMMIT = (
    "6a51ad4ed1d66067c0ac2d3f2c8c3b5de0f5d2ba"
)
E0_COMMIT_CHAIN = {
    "scientific_operator_and_image_source": E0_SCIENTIFIC_CODE_COMMIT,
    "initial_execution_and_config_lock": "c8751b2a933484ca250b2dcf3f80e233e3809cf6",
    "corrected_dev_tree_lock_and_replacement_configmap_basis": "ce99abc066ca55d3d6b75f1005e1d4e6188136f3",
    "final_decode_evaluation_image_and_source": E0_FINAL_EVALUATION_SOURCE_COMMIT,
    "decode_recovery_execution_lock": "a1bb56cf0b315fdf4b03d56eab2e5a91560c3907",
    "final_result_archive": "613958af38fad27e1ea933ccc0dda6d1af5cce89",
}
E0_UNIFIED_EVALUATOR_SHA256 = (
    "bba1a0d3591a00ce62586f055e92875fb01e84c708c42e922493c3aef91a0371"
)
E0_BUILD_PROMPT_SOURCE_SHA256 = (
    "74b7ad7ebfd1df557c3eb4c36075bfe45958c915574a498622f68d287cfb4332"
)
E0_SET_DEFAULT_CHAT_TEMPLATE_SOURCE_SHA256 = (
    "0b5a3e0f9890daf6393f3be73c9b55fc9f93e77ed881c58994523efc9943ccb6"
)
E0_SOURCE_CLOSURE_SHA256 = {
    "rosetta/model/aligner.py": "fe77d72fb7103ca103fe87fa602bed545e0caf6fc75b1741b8736b41e9daf7d8",
    "rosetta/train/dataset_adapters.py": "125c22bd09b2eec0f021f75496ff2a74ef1f795d9b8c0336b70342e6eba08dc0",
    "rosetta/utils/model_loading.py": "d99b45372f1a756cd4e7bf8443c6ab838102e8dfa89f96bdc3b3910ca6532b3a",
    "script/experiment/fpct_e0_runner.py": "d3008d18bd19bb13c8bdc5526412c23bbe64dd3ba0a7ffe799f3f928d9d08f0e",
}
E0_HISTORICAL_EVALUATE_FILE_SHA256 = (
    "1e1d06d79769386a63436cf230cbc9c5ef09d9342b7f060b7238e4a57f94d50d"
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_data(value: Any) -> Any:
    """Convert one materialized dataset row to strict canonical JSON data."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        # canonical_json_bytes rejects NaN/Inf via allow_nan=False.
        return value
    if isinstance(value, Mapping):
        return {str(key): _canonical_data(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_data(child) for child in value]
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if converted is not value:
            return _canonical_data(converted)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if converted is not value:
            return _canonical_data(converted)
    raise TypeError(f"materialized row contains non-canonical value {type(value)!r}")


def raw_full_row_sha256(example: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(_canonical_data(example)))


def canonical_historical_content_sha256(
    question: str, choices: Sequence[str]
) -> str:
    """Mirror the immutable FPCT-1B first-four/pad-to-ten identity."""

    normalize = lambda value: " ".join(str(value).strip().split())
    padded = [
        normalize(choices[index]) if index < min(4, len(choices)) else ""
        for index in range(10)
    ]
    return sha256_bytes(
        canonical_json_bytes({"question": normalize(question), "choices": padded})
    )


def _choice_payload(task: str, example: Mapping[str, Any]) -> tuple[str, list[str], list[str]]:
    """Return the exact full runtime-ordered question/choices/labels."""

    if task == "mmlu-redux":
        question = str(example.get("question", ""))
        if isinstance(example.get("choices"), Sequence) and not isinstance(
            example.get("choices"), (str, bytes)
        ):
            choices = [str(value) for value in example.get("choices", [])]
        else:
            choices = [str(example.get(letter, "")) for letter in "ABCD"]
        formatter_choices = [str(example.get(letter, "")) for letter in "ABCD"]
        if any(formatter_choices) and choices[:4] != formatter_choices:
            raise ValueError("A5 MMLU choices differ from exact formatter A-D fields")
        labels = [chr(65 + index) for index in range(len(choices))]
        return question, choices, labels
    if task == "ai2-arc":
        question = str(example.get("question", ""))
    elif task == "openbookqa":
        question = str(example.get("question_stem", ""))
    else:
        raise ValueError(f"unsupported A5 task: {task}")
    raw = example.get("choices")
    if isinstance(raw, Mapping):
        choices = [str(value) for value in raw.get("text", [])]
        raw_labels = raw.get("label", [])
        labels = [str(value) for value in raw_labels]
    elif isinstance(raw, list):
        choices = [
            str(value.get("text", "")) if isinstance(value, Mapping) else str(value)
            for value in raw
        ]
        labels = [
            str(value.get("label", chr(65 + index)))
            if isinstance(value, Mapping)
            else chr(65 + index)
            for index, value in enumerate(raw)
        ]
    else:
        raise ValueError("A5 materialized row has no ordered choice collection")
    if len(labels) != len(choices):
        raise ValueError("A5 materialized choice labels/text lengths differ")
    return question, choices, labels


def full_question_choices(
    task: str, example: Mapping[str, Any]
) -> tuple[str, list[str], list[str]]:
    question, choices, labels = _choice_payload(task, example)
    if not question:
        raise ValueError("A5 materialized question is empty")
    if len(choices) < 4:
        raise ValueError("A5 materialized row has fewer than four choices")
    expected_labels = [chr(65 + index) for index in range(len(choices))]
    if labels != expected_labels:
        raise ValueError("A5 production choice labels/order are not canonical A..")
    return question, choices, labels


def historical_projected_example(
    task: str, example: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a deep JSON projection whose renderer sees exactly first four."""

    projected = _canonical_data(example)
    if not isinstance(projected, dict):
        raise TypeError("A5 materialized example is not an object")
    question, choices, labels = full_question_choices(task, example)
    if task in {"ai2-arc", "openbookqa"}:
        raw = projected.get("choices")
        if isinstance(raw, dict):
            raw["text"] = choices[:4]
            raw["label"] = labels[:4]
        elif isinstance(raw, list):
            projected["choices"] = raw[:4]
        else:  # pragma: no cover - full_question_choices already rejects this
            raise ValueError("A5 projected example has no choice collection")
    elif task == "mmlu-redux":
        if isinstance(projected.get("choices"), list):
            projected["choices"] = choices[:4]
        # The exact E0 MMLU formatter consumes A/B/C/D fields.  Retain those
        # fields byte-for-byte and reject any unexpected suffix above.
        if len(choices) != 4:
            raise ValueError("A5 MMLU production row unexpectedly exceeds A-D")
    # Prove the projection did not mutate the raw question or first four.
    historical_question, historical_choices, historical_labels = full_question_choices(
        task, projected
    )
    if (
        historical_question != question
        or historical_choices != choices[:4]
        or historical_labels != labels[:4]
    ):
        raise ValueError("A5 first-four projection changed question/choice identity")
    return projected


def classify_prompt_relation(
    *,
    historical_question: str,
    historical_choices: Sequence[str],
    historical_labels: Sequence[str],
    production_question: str,
    production_choices: Sequence[str],
    production_labels: Sequence[str],
    gold_answer: str,
    historical_rendered_prompt: str,
    production_rendered_prompt: str,
    historical_alignment_sha256: str,
    production_alignment_sha256: str,
) -> tuple[str, bool]:
    """Classify the only two allowed A5 relations or fail closed."""

    if gold_answer not in GOLD_ANSWERS:
        raise ValueError("A5 gold answer is outside A-D")
    if len(historical_choices) != 4 or len(historical_labels) != 4:
        raise ValueError("A5 historical projection is not exactly first-four")
    if historical_question != production_question:
        raise ValueError("A5 historical/production question text changed")
    if list(historical_choices) != list(production_choices[:4]):
        raise ValueError("A5 production first-four choice text/order changed")
    if list(historical_labels) != list(production_labels[:4]):
        raise ValueError("A5 production first-four choice labels/order changed")
    if list(historical_labels) != list("ABCD"):
        raise ValueError("A5 historical first-four labels are not A-D")
    if len(production_choices) != len(production_labels):
        raise ValueError("A5 production choice label/text lengths differ")
    if len(production_choices) == 4:
        if historical_rendered_prompt != production_rendered_prompt:
            raise ValueError("A5 four-choice renderer bytes differ")
        if historical_alignment_sha256 != production_alignment_sha256:
            raise ValueError("A5 four-choice alignment differs")
        return HISTORICAL_EXACT, False
    if len(production_choices) > 4:
        if historical_rendered_prompt == production_rendered_prompt:
            raise ValueError("A5 extra choices did not affect production prompt bytes")
        return EXTRA_CHOICES_ONLY, True
    raise ValueError("A5 production row has fewer choices than historical projection")


def dual_anchor_record(
    *,
    task: str,
    content_group_sha256: str,
    sample_key_sha256: str,
    source_row_id: str,
    example: Mapping[str, Any],
    gold_answer: str,
    historical_rendered_prompt: str,
    historical_alignment_sha256: str,
    production_rendered_prompt: str,
    production_alignment_sha256: str,
    historical_prompt_token_count: int,
    production_prompt_token_count: int,
    production_certified_parent_count: int,
    production_logical_row_count: int,
    production_physical_chunk_count: int,
    historical_content_sha256: str,
) -> dict[str, Any]:
    projected = historical_projected_example(task, example)
    historical_question, historical_choices, historical_labels = full_question_choices(
        task, projected
    )
    production_question, production_choices, production_labels = full_question_choices(
        task, example
    )
    recomputed_historical_content = canonical_historical_content_sha256(
        historical_question, historical_choices
    )
    if not (
        recomputed_historical_content
        == historical_content_sha256
        == content_group_sha256
    ):
        raise ValueError("A5 historical first-four content-group anchor changed")
    relation, choice_only = classify_prompt_relation(
        historical_question=historical_question,
        historical_choices=historical_choices,
        historical_labels=historical_labels,
        production_question=production_question,
        production_choices=production_choices,
        production_labels=production_labels,
        gold_answer=gold_answer,
        historical_rendered_prompt=historical_rendered_prompt,
        production_rendered_prompt=production_rendered_prompt,
        historical_alignment_sha256=historical_alignment_sha256,
        production_alignment_sha256=production_alignment_sha256,
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (
            historical_prompt_token_count,
            production_prompt_token_count,
            production_certified_parent_count,
            production_logical_row_count,
            production_physical_chunk_count,
        )
    ):
        raise ValueError("A5 census contains an invalid nonnegative count")
    first4_payload = {
        "question": historical_question,
        "choices": historical_choices,
    }
    first4_sha = sha256_bytes(canonical_json_bytes(first4_payload))
    return {
        "schema_version": 7,
        "protocol_id": A5_PROTOCOL_ID,
        "artifact_type": "a5_prompt_census_record",
        "task": task,
        "content_group_sha256": content_group_sha256,
        "sample_key_sha256": sample_key_sha256,
        "source_row_id": str(source_row_id),
        "historical_choice_count": 4,
        "production_choice_count": len(production_choices),
        "raw_choice_labels": production_labels,
        "gold_answer": gold_answer,
        "historical_first4_question_choices_sha256": first4_sha,
        "historical_rendered_prompt_sha256": sha256_bytes(
            historical_rendered_prompt.encode("utf-8")
        ),
        "historical_alignment_sha256": historical_alignment_sha256,
        "raw_full_row_sha256": raw_full_row_sha256(example),
        "production_rendered_prompt_sha256": sha256_bytes(
            production_rendered_prompt.encode("utf-8")
        ),
        "production_alignment_sha256": production_alignment_sha256,
        "historical_prompt_token_count": historical_prompt_token_count,
        "production_prompt_token_count": production_prompt_token_count,
        "production_certified_parent_count": production_certified_parent_count,
        "production_logical_row_count": production_logical_row_count,
        "production_physical_chunk_count": production_physical_chunk_count,
        "prompt_relation": relation,
        "choice_difference_only": choice_only,
    }


def _count_distribution(values: Sequence[int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items(), key=lambda pair: int(pair[0])))


def _distribution_summary(
    records: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    ordered = sorted(
        (int(record[field]), str(record["content_group_sha256"]))
        for record in records
    )
    if not ordered:
        return {
            "count": 0,
            "sum": 0,
            "min": 0,
            "p50": 0,
            "p95": 0,
            "max": 0,
            "argmax_content_group_sha256": None,
        }
    count = len(ordered)

    def nearest_rank(numerator: int, denominator: int) -> int:
        index = (numerator * count + denominator - 1) // denominator - 1
        return ordered[max(index, 0)][0]

    maximum = ordered[-1][0]
    return {
        "count": count,
        "sum": sum(value for value, _ in ordered),
        "min": ordered[0][0],
        "p50": nearest_rank(50, 100),
        "p95": nearest_rank(95, 100),
        "max": maximum,
        "argmax_content_group_sha256": min(
            group for value, group in ordered if value == maximum
        ),
    }


def summarize_dual_anchor_census(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_task_counts: Mapping[str, int] = EXPECTED_TASK_COUNTS,
) -> dict[str, Any]:
    """Fail-closed full-population reduction in canonical task/group order."""

    task_rank = {task: index for index, task in enumerate(TASK_ORDER)}
    ordered = sorted(
        (dict(record) for record in records),
        key=lambda row: (
            task_rank.get(str(row.get("task")), len(task_rank)),
            str(row.get("content_group_sha256", "")),
            str(row.get("sample_key_sha256", "")),
        ),
    )
    if list(records) != ordered:
        raise ValueError("A5 census rows are not in canonical task/group/sample order")
    expected_total = sum(int(value) for value in expected_task_counts.values())
    if len(ordered) != expected_total:
        raise ValueError("A5 census does not contain the exact frozen population")
    group_ids = [str(row.get("content_group_sha256", "")) for row in ordered]
    sample_ids = [str(row.get("sample_key_sha256", "")) for row in ordered]
    if len(set(group_ids)) != len(group_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("A5 census contains duplicate group/sample identity")
    counts = {task: 0 for task in TASK_ORDER}
    for row in ordered:
        task = row.get("task")
        if task not in counts:
            raise ValueError("A5 census contains an unexpected task")
        counts[str(task)] += 1
        relation = row.get("prompt_relation")
        if relation not in PROMPT_RELATIONS:
            raise ValueError("A5 census contains an unclassified prompt difference")
        if bool(row.get("choice_difference_only")) != (
            relation == EXTRA_CHOICES_ONLY
        ):
            raise ValueError("A5 census relation/choice-only flag disagree")
        if row.get("gold_answer") not in GOLD_ANSWERS:
            raise ValueError("A5 census contains a gold answer outside A-D")
        if row.get("historical_choice_count") != 4:
            raise ValueError("A5 census historical choice count changed")
        labels = row.get("raw_choice_labels")
        production_count = row.get("production_choice_count")
        if (
            not isinstance(labels, list)
            or not isinstance(production_count, int)
            or labels != [chr(65 + index) for index in range(production_count)]
        ):
            raise ValueError("A5 census production choice order is invalid")
    if counts != dict(expected_task_counts):
        raise ValueError("A5 census task counts differ from frozen membership")
    by_task: dict[str, Any] = {}
    affected: list[str] = []
    for task in TASK_ORDER:
        members = [row for row in ordered if row["task"] == task]
        task_affected = [
            row for row in members if row["prompt_relation"] == EXTRA_CHOICES_ONLY
        ]
        affected.extend(str(row["content_group_sha256"]) for row in task_affected)
        by_task[task] = {
            "group_count": len(members),
            "affected_group_count": len(task_affected),
            "unaffected_group_count": len(members) - len(task_affected),
            "historical_choice_count_distribution": _count_distribution(
                [int(row["historical_choice_count"]) for row in members]
            ),
            "production_choice_count_distribution": _count_distribution(
                [int(row["production_choice_count"]) for row in members]
            ),
            "production_prompt_token_count_distribution": _count_distribution(
                [int(row["production_prompt_token_count"]) for row in members]
            ),
            "production_certified_parent_count_distribution": _count_distribution(
                [int(row["production_certified_parent_count"]) for row in members]
            ),
            "production_logical_row_count_distribution": _count_distribution(
                [int(row["production_logical_row_count"]) for row in members]
            ),
            "production_physical_chunk_count_distribution": _count_distribution(
                [int(row["production_physical_chunk_count"]) for row in members]
            ),
        }
    return {
        "schema_version": 7,
        "protocol_id": A5_PROTOCOL_ID,
        "artifact_type": "a5_dual_anchor_census_summary",
        "population": "e0_design",
        "group_count": len(ordered),
        "task_counts": counts,
        "by_task": by_task,
        "affected_content_group_sha256": affected,
        "unexpected_prompt_difference_count": 0,
        "missing_rows": 0,
        "duplicate_rows": 0,
        "historical_to_runtime_mapping_complete": True,
        "records": ordered,
        "semantic_sha256": sha256_bytes(canonical_json_bytes(ordered)),
    }


def build_census_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    execution_sha: str,
    run_uid: str,
    record_artifact: Mapping[str, Any],
    expected_task_counts: Mapping[str, int] = EXPECTED_TASK_COUNTS,
) -> dict[str, Any]:
    """Build the strict A5 schema manifest after semantic census validation."""

    summary = summarize_dual_anchor_census(
        records, expected_task_counts=expected_task_counts
    )
    ordered = list(summary["records"])
    affected_task_counts = {
        task: int(summary["by_task"][task]["affected_group_count"])
        for task in TASK_ORDER
    }
    unaffected_task_counts = {
        task: int(summary["by_task"][task]["unaffected_group_count"])
        for task in TASK_ORDER
    }
    anchor_fields = (
        "task",
        "content_group_sha256",
        "sample_key_sha256",
        "historical_first4_question_choices_sha256",
        "historical_rendered_prompt_sha256",
        "historical_alignment_sha256",
        "raw_full_row_sha256",
        "production_rendered_prompt_sha256",
        "production_alignment_sha256",
        "prompt_relation",
    )
    artifact = dict(record_artifact)
    expected_artifact_fields = {"relative_path", "sha256", "bytes", "row_count"}
    if set(artifact) != expected_artifact_fields or artifact["row_count"] != len(
        ordered
    ):
        raise ValueError("A5 census record artifact descriptor is invalid")
    return {
        "schema_version": 7,
        "protocol_id": A5_PROTOCOL_ID,
        "artifact_type": "a5_prompt_census_manifest",
        "status": "COMPLETE_PRE_MODEL_CENSUS",
        "execution_sha": execution_sha,
        "run_uid": run_uid,
        "population_count": len(ordered),
        "task_counts": dict(summary["task_counts"]),
        "affected_task_counts": affected_task_counts,
        "unaffected_task_counts": unaffected_task_counts,
        "historical_choice_count_distribution": _count_distribution(
            [int(row["historical_choice_count"]) for row in ordered]
        ),
        "production_choice_count_distribution": _count_distribution(
            [int(row["production_choice_count"]) for row in ordered]
        ),
        "affected_content_group_sha256": list(
            summary["affected_content_group_sha256"]
        ),
        "historical_to_production_anchor_map": [
            {field: row[field] for field in anchor_fields} for row in ordered
        ],
        "production_prompt_token_count_distribution": _distribution_summary(
            ordered, "production_prompt_token_count"
        ),
        "production_certified_parent_count_distribution": _distribution_summary(
            ordered, "production_certified_parent_count"
        ),
        "production_logical_row_count_distribution": _distribution_summary(
            ordered, "production_logical_row_count"
        ),
        "production_physical_chunk_count_distribution": _distribution_summary(
            ordered, "production_physical_chunk_count"
        ),
        "unexpected_prompt_difference_count": 0,
        "missing_row_count": 0,
        "duplicate_row_count": 0,
        "canonical_semantic_stream_sha256": summary["semantic_sha256"],
        "artifacts": [artifact],
    }


def _ast_source_sha256(path: Path, *, function_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        raise ValueError(f"A5 renderer source has {len(matches)} {function_name} functions")
    segment = ast.get_source_segment(source, matches[0])
    if not isinstance(segment, str):
        raise ValueError("A5 cannot extract exact prompt-builder source")
    return sha256_bytes(segment.encode("utf-8"))


def attest_e0_renderer_identity(repo_root: Path) -> dict[str, Any]:
    """Prove the active A5 renderer functions equal the immutable E0 oracle."""

    evaluator = repo_root / "script/evaluation/unified_evaluator.py"
    prompt_builder = repo_root / "rosetta/utils/evaluate.py"
    evaluator_sha = sha256_file(evaluator)
    build_prompt_sha = _ast_source_sha256(
        prompt_builder, function_name="build_prompt"
    )
    set_template_sha = _ast_source_sha256(
        prompt_builder, function_name="set_default_chat_template"
    )
    if evaluator_sha != E0_UNIFIED_EVALUATOR_SHA256:
        raise ValueError("A5 UnifiedEvaluator differs from the exact E0 renderer")
    if build_prompt_sha != E0_BUILD_PROMPT_SOURCE_SHA256:
        raise ValueError("A5 build_prompt differs from the exact E0 prompt builder")
    if set_template_sha != E0_SET_DEFAULT_CHAT_TEMPLATE_SOURCE_SHA256:
        raise ValueError("A5 set_default_chat_template differs from exact E0")
    closure = {
        relative: sha256_file(repo_root / relative)
        for relative in E0_SOURCE_CLOSURE_SHA256
    }
    if closure != E0_SOURCE_CLOSURE_SHA256:
        raise ValueError("A5 E0 renderer/alignment source closure differs")
    return {
        "e0_final_evaluation_source_commit": E0_FINAL_EVALUATION_SOURCE_COMMIT,
        "e0_scientific_code_commit": E0_SCIENTIFIC_CODE_COMMIT,
        "e0_commit_chain": dict(E0_COMMIT_CHAIN),
        "unified_evaluator_sha256": evaluator_sha,
        "historical_evaluate_file_sha256": E0_HISTORICAL_EVALUATE_FILE_SHA256,
        "active_evaluate_file_sha256": sha256_file(prompt_builder),
        "build_prompt_source_sha256": build_prompt_sha,
        "set_default_chat_template_source_sha256": set_template_sha,
        "source_closure_sha256": closure,
        "renderer_source_identity_attested": True,
        "production_renderer_exactly_attested": False,
        "production_renderer_exact_attestation_pending": (
            "all_326_full_row_prechat_and_dual_tokenizer_render_replays"
        ),
    }
