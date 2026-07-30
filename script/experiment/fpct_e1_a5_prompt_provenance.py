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
A5R1_PROTOCOL_ID = "fpct_e1_mechanism_audit_v8_a5r1_hash_domains"
A5R1_AMENDMENT_ID = "APPROVED_PROSPECTIVE_AMENDMENT_E1_A5R1_HASH_DOMAINS"
A5R2_PROTOCOL_ID = "fpct_e1_mechanism_audit_v9_a5r2_choice_cardinality"
A5R2_AMENDMENT_ID = (
    "APPROVED_PROSPECTIVE_AMENDMENT_E1_A5R2_CHOICE_CARDINALITY_RECOVERY"
)
HISTORICAL_EXACT = "EXACT_HISTORICAL_AND_PRODUCTION_MATCH"
EXTRA_CHOICES_ONLY = "EXTRA_CHOICES_ONLY"
PROMPT_RELATIONS = frozenset((HISTORICAL_EXACT, EXTRA_CHOICES_ONLY))
TASK_ORDER = ("ai2-arc", "openbookqa", "mmlu-redux")
EXPECTED_TASK_COUNTS = {"ai2-arc": 128, "openbookqa": 70, "mmlu-redux": 128}
GOLD_ANSWERS = frozenset(("A", "B", "C", "D"))
A5R2_ROOT_CAUSE_ACTIONS = {
    "DIRECT_VALID": "NONE",
    "GENUINE_ARC_LOW_CARDINALITY": "VARIABLE_CARDINALITY_CONTRACT_ONLY",
    "HF_FEATURE_NORMALIZATION_ONLY": "PROVENANCE_CANONICAL_VIEW_ONLY",
    "DESCRIPTOR_INDEX_MISMATCH": "BLOCK",
    "UNSUPPORTED_FEATURE_REPRESENTATION": "BLOCK",
    "GENUINE_TASK_INVALID_CARDINALITY": "BLOCK",
    "UNEXPLAINED_OTHER": "BLOCK",
}
A5R2_REQUIRED_ZERO_ROOT_CAUSES = frozenset(
    (
        "DESCRIPTOR_INDEX_MISMATCH",
        "UNSUPPORTED_FEATURE_REPRESENTATION",
        "GENUINE_TASK_INVALID_CARDINALITY",
        "UNEXPLAINED_OTHER",
    )
)


def _validated_protocol_header(
    schema_version: int, protocol_id: str,
) -> tuple[int, str]:
    """Allow immutable v7/v8 behavior plus the explicit v9 successor."""

    supported = {
        (7, A5_PROTOCOL_ID),
        (8, A5R1_PROTOCOL_ID),
        (9, A5R2_PROTOCOL_ID),
    }
    header = (schema_version, protocol_id)
    if header not in supported:
        raise ValueError("unsupported A5 prompt provenance protocol header")
    return header

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


def _finite_choice_list_v9(value: Any, *, name: str) -> tuple[list[Any], bool]:
    """Return one finite list via a single explicit representation path."""

    if isinstance(value, (str, bytes)):
        raise ValueError(f"A5R2 {name} cannot be string/bytes")
    if isinstance(value, (list, tuple)):
        return list(value), False
    converters = [
        converter_name
        for converter_name in ("to_pylist", "as_py", "tolist")
        if callable(getattr(value, converter_name, None))
    ]
    if len(converters) != 1:
        raise ValueError(f"A5R2 {name} has no unique finite feature conversion")
    converted = getattr(value, converters[0])()
    if not isinstance(converted, (list, tuple)):
        raise ValueError(f"A5R2 {name} feature conversion is not a finite sequence")
    return list(converted), True


def _strict_labeled_choice_collection_v9(
    raw: Any,
) -> tuple[list[str], list[str], str, bool]:
    """Parse one labeled choice container without guessing between shapes.

    The production datasets use either a mapping of parallel ``text``/``label``
    sequences or a list of per-choice mappings.  Mixed lists, missing labels,
    scalar pseudo-sequences and unequal parallel lengths are rejected rather
    than normalized into a superficially plausible row.
    """

    if isinstance(raw, Mapping):
        if "text" not in raw or "label" not in raw:
            raise ValueError("A5R2 mapping choices require text and label sequences")
        texts_raw, labels_raw = raw["text"], raw["label"]
        texts, text_normalized = _finite_choice_list_v9(
            texts_raw, name="mapping choice text"
        )
        labels, label_normalized = _finite_choice_list_v9(
            labels_raw, name="mapping choice label"
        )
        if any(not isinstance(value, str) for value in texts) or any(
            not isinstance(value, str) for value in labels
        ):
            raise ValueError("A5R2 mapping choice text/label must be strings")
        choices = list(texts)
        source_labels = list(labels)
        normalized = text_normalized or label_normalized
        container_kind = (
            "feature_backed_sequence_to_list"
            if normalized
            else "mapping_of_sequences"
        )
    else:
        normalized_raw = False
        if not isinstance(raw, list):
            raw, normalized_raw = _finite_choice_list_v9(
                raw, name="choice collection"
            )
        if not raw or not all(isinstance(value, Mapping) for value in raw):
            raise ValueError(
                "A5R2 list choices must be a nonempty list of mappings"
            )
        if any("text" not in value or "label" not in value for value in raw):
            raise ValueError("A5R2 list choice mappings require text and label")
        if any(
            not isinstance(value["text"], str)
            or not isinstance(value["label"], str)
            for value in raw
        ):
            raise ValueError("A5R2 list choice text/label must be strings")
        choices = [value["text"] for value in raw]
        source_labels = [value["label"] for value in raw]
        normalized = normalized_raw
        container_kind = (
            "feature_backed_sequence_to_list"
            if normalized
            else "sequence_of_mappings"
        )
    if len(choices) != len(source_labels):
        raise ValueError("A5R2 materialized choice labels/text lengths differ")
    if any(not value.strip() for value in choices):
        raise ValueError("A5R2 materialized choice text is empty")
    if any(not value.strip() for value in source_labels):
        raise ValueError("A5R2 materialized source choice label is empty")
    if len(set(source_labels)) != len(source_labels):
        raise ValueError("A5R2 materialized source choice labels are not unique")
    return choices, source_labels, container_kind, normalized


def choice_payload_v9(
    task: str,
    example: Mapping[str, Any],
    *,
    enforce_task_cardinality: bool = True,
) -> dict[str, Any]:
    """Return the v9 runtime choice geometry without reading an answer field.

    Source labels identify the dataset's choice records.  Runtime ordinal
    labels identify what ``UnifiedEvaluator`` actually prints (A, B, ...).
    They are intentionally separate even when their values happen to match.
    The returned production choice list is complete and is never padded or
    truncated.
    """

    if task == "ai2-arc":
        question = example.get("question", "")
        (
            choices,
            source_labels,
            container_kind,
            feature_normalized,
        ) = _strict_labeled_choice_collection_v9(example.get("choices"))
        if enforce_task_cardinality and len(choices) < 2:
            raise ValueError("A5R2 ARC row has fewer than two choices")
    elif task == "openbookqa":
        question = example.get("question_stem", "")
        (
            choices,
            source_labels,
            container_kind,
            feature_normalized,
        ) = _strict_labeled_choice_collection_v9(example.get("choices"))
        if enforce_task_cardinality and len(choices) != 4:
            raise ValueError("A5R2 OpenBookQA row is not exactly four-choice")
    elif task == "mmlu-redux":
        question = example.get("question", "")
        raw, feature_normalized = _finite_choice_list_v9(
            example.get("choices"), name="MMLU choices"
        )
        if any(isinstance(value, Mapping) for value in raw):
            raise ValueError("A5R2 MMLU choices must be one scalar sequence")
        if any(not isinstance(value, str) for value in raw):
            raise ValueError("A5R2 MMLU choice text must be strings")
        choices = list(raw)
        if enforce_task_cardinality and len(choices) != 4:
            raise ValueError("A5R2 MMLU row is not exactly four-choice")
        if any(not value.strip() for value in choices):
            raise ValueError("A5R2 materialized choice text is empty")
        # MMLU carries a scalar choice sequence rather than dataset-provided
        # source labels.  Its structural source-label domain is therefore the
        # same deterministic ordinal domain as the unchanged runtime
        # formatter.  This must also hold for a parseable task-invalid row:
        # the audit has to persist that row and reach a complete mechanical
        # BLOCK lock instead of crashing during summary reduction because a
        # hard-coded four-label vector no longer matches the observed count.
        source_labels = [chr(65 + index) for index in range(len(choices))]
        container_kind = (
            "feature_backed_sequence_to_list"
            if feature_normalized
            else "plain_mmlu_sequence"
        )
    else:
        raise ValueError(f"unsupported A5R2 task: {task}")
    if not isinstance(question, str) or not question:
        raise ValueError("A5R2 materialized question must be a nonempty string")
    if len(choices) > 26:
        raise ValueError("A5R2 choice count exceeds ordinal label alphabet")
    runtime_labels = [chr(65 + index) for index in range(len(choices))]
    return {
        "question": question,
        "production_choices": choices,
        "source_choice_labels": source_labels,
        "runtime_ordinal_labels": runtime_labels,
        "choice_container_kind": container_kind,
        "feature_container_normalized": feature_normalized,
        "representation_status": (
            "HF_FEATURE_NORMALIZATION_REQUIRED"
            if feature_normalized
            else "DIRECT"
        ),
        "production_choice_count": len(choices),
        "historical_choice_count": min(4, len(choices)),
    }


def validate_gold_answer_v9(
    gold_answer: str,
    *,
    source_choice_labels: Sequence[str],
    runtime_ordinal_labels: Sequence[str],
) -> None:
    """Require a historical A-D answer that names an actual runtime choice."""

    if gold_answer not in GOLD_ANSWERS:
        raise ValueError("A5R2 gold answer is outside A-D")
    if gold_answer not in runtime_ordinal_labels:
        raise ValueError("A5R2 gold answer has no runtime ordinal choice")
    # Source labels are structural provenance only.  ARC may use labels such
    # as 1/2/3/4 while the unchanged E0 formatter emits runtime A/B/C/D.
    # Therefore source labels must never be treated as gold-answer authority.


def full_question_choices_v9(
    task: str, example: Mapping[str, Any]
) -> tuple[str, list[str], list[str], list[str]]:
    """Public production parser: full choices, source labels, runtime labels."""

    payload = choice_payload_v9(task, example)
    return (
        str(payload["question"]),
        list(payload["production_choices"]),
        list(payload["source_choice_labels"]),
        list(payload["runtime_ordinal_labels"]),
    )


def historical_projected_example_v9(
    task: str, example: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the historical prompt view from the parsed canonical payload.

    Do not deep-copy the materialized row: feature-backed choice containers may
    expose their finite representation through ``to_pylist``/``as_py`` rather
    than ``tolist``, and outcome fields do not belong in this prompt view.
    The production example itself is never mutated.
    """

    payload = choice_payload_v9(task, example)
    historical_count = int(payload["historical_choice_count"])
    choices = list(payload["production_choices"][:historical_count])
    labels = list(payload["source_choice_labels"][:historical_count])
    if task == "ai2-arc":
        projected: dict[str, Any] = {
            "question": payload["question"],
            "choices": {"text": choices, "label": labels},
        }
    elif task == "openbookqa":
        projected = {
            "question_stem": payload["question"],
            "choices": {"text": choices, "label": labels},
        }
    elif task == "mmlu-redux":
        projected = {"question": payload["question"], "choices": choices}
    else:  # pragma: no cover - choice_payload_v9 already rejects this
        raise ValueError(f"unsupported A5R2 task: {task}")
    # OpenBookQA and MMLU are exactly four; ARC is the only task for which the
    # historical projection can differ from the full production row.
    projected_payload = choice_payload_v9(task, projected)
    if (
        projected_payload["question"] != payload["question"]
        or projected_payload["production_choices"]
        != payload["production_choices"][:historical_count]
        or projected_payload["source_choice_labels"]
        != payload["source_choice_labels"][:historical_count]
    ):
        raise ValueError("A5R2 projection changed question/choice identity")
    return projected


def classify_prompt_relation_v9(
    *,
    historical_payload: Mapping[str, Any],
    production_payload: Mapping[str, Any],
    gold_answer: str,
    historical_rendered_prompt: str,
    production_rendered_prompt: str,
    historical_alignment_sha256: str,
    production_alignment_sha256: str,
) -> tuple[str, bool]:
    """Classify v9 exact/suffix-only relations with variable ARC cardinality."""

    production_choices = list(production_payload["production_choices"])
    historical_choices = list(historical_payload["production_choices"])
    production_runtime = list(production_payload["runtime_ordinal_labels"])
    historical_runtime = list(historical_payload["runtime_ordinal_labels"])
    production_source = list(production_payload["source_choice_labels"])
    historical_source = list(historical_payload["source_choice_labels"])
    validate_gold_answer_v9(
        gold_answer,
        source_choice_labels=production_source,
        runtime_ordinal_labels=production_runtime,
    )
    expected_historical_count = min(4, len(production_choices))
    if len(historical_choices) != expected_historical_count:
        raise ValueError("A5R2 historical choice count is not min(4,n)")
    if historical_payload["question"] != production_payload["question"]:
        raise ValueError("A5R2 historical/production question text changed")
    if historical_choices != production_choices[:expected_historical_count]:
        raise ValueError("A5R2 historical/production choice text/order changed")
    if historical_source != production_source[:expected_historical_count]:
        raise ValueError("A5R2 historical/production source labels changed")
    if historical_runtime != production_runtime[:expected_historical_count]:
        raise ValueError("A5R2 historical/runtime ordinal labels changed")
    if len(production_choices) <= 4:
        if historical_rendered_prompt != production_rendered_prompt:
            raise ValueError("A5R2 <=4-choice renderer bytes differ")
        if historical_alignment_sha256 != production_alignment_sha256:
            raise ValueError("A5R2 <=4-choice alignment differs")
        return HISTORICAL_EXACT, False
    if historical_rendered_prompt == production_rendered_prompt:
        raise ValueError("A5R2 suffix choices did not affect production prompt bytes")
    return EXTRA_CHOICES_ONLY, True


A5R2_DESCRIPTOR_FIELDS = (
    "dataset_id",
    "dataset_config",
    "dataset_split",
    "evaluation_question_id",
    "non_outcome_native_row_id",
    "feature_schema_sha256",
)
A5R2_DESCRIPTOR_IDENTITY_FIELDS = (
    "dataset_id",
    "dataset_config",
    "dataset_split",
    "evaluation_question_id",
    "non_outcome_native_row_id",
)


def _lower_sha256_v9(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"A5R2 {name} is not lowercase SHA256")
    return value


def _descriptor_pair_v9(descriptor: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(descriptor) != {"expected", "observed"}:
        raise ValueError("A5R2 descriptor requires exact expected/observed views")
    expected, observed = descriptor["expected"], descriptor["observed"]
    if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
        raise ValueError("A5R2 descriptor views must be mappings")
    required = set(A5R2_DESCRIPTOR_FIELDS)
    if set(expected) != required or set(observed) != required:
        raise ValueError("A5R2 descriptor view fields differ from the allowlist")
    expected_dict, observed_dict = dict(expected), dict(observed)
    for name, view in (("expected", expected_dict), ("observed", observed_dict)):
        for field in ("dataset_id", "dataset_config", "dataset_split"):
            if not isinstance(view[field], str) or not view[field]:
                raise ValueError(f"A5R2 {name} descriptor {field} is empty")
        if view["dataset_split"] != "test":
            raise ValueError(f"A5R2 {name} descriptor split is not test")
        question_id = view["evaluation_question_id"]
        if (
            isinstance(question_id, bool)
            or not isinstance(question_id, int)
            or question_id < 0
        ):
            raise ValueError(f"A5R2 {name} descriptor question index is invalid")
        _lower_sha256_v9(
            view["feature_schema_sha256"], name=f"{name} feature schema"
        )
        native = view["non_outcome_native_row_id"]
        if native is not None and not isinstance(native, (str, int)):
            raise ValueError(f"A5R2 {name} native row ID is not scalar/null")
    return expected_dict, observed_dict


def choice_audit_projection_v9(
    task: str,
    descriptor: Mapping[str, Any],
    example: Mapping[str, Any],
) -> dict[str, Any]:
    """Create ephemeral formatter views and a persistable label-free projection."""

    expected_descriptor, observed_descriptor = _descriptor_pair_v9(descriptor)
    payload = choice_payload_v9(task, example, enforce_task_cardinality=False)
    raw_choices = example.get("choices")
    if task == "ai2-arc":
        raw_formatter_example = {"question": payload["question"], "choices": raw_choices}
        canonical_formatter_example = {
            "question": payload["question"],
            "choices": {
                "text": list(payload["production_choices"]),
                "label": list(payload["source_choice_labels"]),
            },
        }
    elif task == "openbookqa":
        raw_formatter_example = {
            "question_stem": payload["question"],
            "choices": raw_choices,
        }
        canonical_formatter_example = {
            "question_stem": payload["question"],
            "choices": {
                "text": list(payload["production_choices"]),
                "label": list(payload["source_choice_labels"]),
            },
        }
    else:
        raw_formatter_example = {"question": payload["question"], "choices": raw_choices}
        canonical_formatter_example = {
            "question": payload["question"],
            "choices": list(payload["production_choices"]),
        }
    label_free_projection = {
        "task": task,
        "question": payload["question"],
        "production_choices": list(payload["production_choices"]),
        "source_choice_labels": list(payload["source_choice_labels"]),
        "runtime_ordinal_labels": list(payload["runtime_ordinal_labels"]),
        "choice_container_kind": payload["choice_container_kind"],
    }
    expected_descriptor_sha = sha256_bytes(canonical_json_bytes(expected_descriptor))
    observed_descriptor_sha = sha256_bytes(canonical_json_bytes(observed_descriptor))
    persisted = {
        "choice_container_kind": payload["choice_container_kind"],
        "feature_container_normalization_status": (
            "HF_FEATURE_NORMALIZATION_REQUIRED"
            if payload["feature_container_normalized"]
            else "DIRECT"
        ),
        "representation_status": payload["representation_status"],
        "historical_choice_count": int(payload["historical_choice_count"]),
        "production_choice_count": int(payload["production_choice_count"]),
        "source_choice_labels": list(payload["source_choice_labels"]),
        "runtime_ordinal_labels": list(payload["runtime_ordinal_labels"]),
        "observed_content_group_sha256": canonical_historical_content_sha256(
            str(payload["question"]), list(payload["production_choices"])
        ),
        "choice_projection_sha256": sha256_bytes(
            canonical_json_bytes(label_free_projection)
        ),
        "expected_descriptor": expected_descriptor,
        "observed_descriptor": observed_descriptor,
        "expected_descriptor_sha256": expected_descriptor_sha,
        "observed_descriptor_sha256": observed_descriptor_sha,
        # Feature/container SHA is observed structural provenance, not an
        # independent frozen row identity.  Identity uses only the exact
        # dataset materialization coordinates; the completed verifier binds
        # those coordinates to the independently loaded frozen manifests.
        "descriptor_provenance_match": all(
            expected_descriptor[field] == observed_descriptor[field]
            for field in A5R2_DESCRIPTOR_IDENTITY_FIELDS
        ),
        "forbidden_outcome_field_access_count": 0,
    }
    return {
        "schema_version": 9,
        "protocol_id": A5R2_PROTOCOL_ID,
        "artifact_type": "a5r2_choice_audit_projection",
        # These two allowlisted views are ephemeral and must never be passed to
        # a persisted artifact builder.
        "raw_formatter_example": raw_formatter_example,
        "canonical_formatter_example": canonical_formatter_example,
        "persisted": persisted,
    }


def _cardinality_status_v9(task: str, count: int) -> str:
    valid = (task == "ai2-arc" and count >= 2) or (
        task in {"openbookqa", "mmlu-redux"} and count == 4
    )
    return "TASK_VALID" if valid else "TASK_INVALID"


def _taxonomy_v9(
    *,
    task: str,
    count: int,
    identity_status: str,
    representation_status: str,
    cardinality_status: str,
    formatter_byte_parity: bool,
    legacy_choice_count: int | None,
) -> tuple[str, str]:
    if identity_status != "MATCH":
        root_cause = "DESCRIPTOR_INDEX_MISMATCH"
    elif representation_status == "UNSUPPORTED":
        root_cause = "UNSUPPORTED_FEATURE_REPRESENTATION"
    elif cardinality_status != "TASK_VALID":
        root_cause = "GENUINE_TASK_INVALID_CARDINALITY"
    elif representation_status == "HF_FEATURE_NORMALIZATION_REQUIRED":
        if formatter_byte_parity and legacy_choice_count != count:
            root_cause = "HF_FEATURE_NORMALIZATION_ONLY"
        else:
            root_cause = "UNEXPLAINED_OTHER"
    elif (
        representation_status == "DIRECT"
        and formatter_byte_parity
        and task == "ai2-arc"
        and count in {2, 3}
    ):
        root_cause = "GENUINE_ARC_LOW_CARDINALITY"
    elif representation_status == "DIRECT" and formatter_byte_parity:
        root_cause = "DIRECT_VALID"
    else:
        root_cause = "UNEXPLAINED_OTHER"
    return root_cause, A5R2_ROOT_CAUSE_ACTIONS[root_cause]


def choice_cardinality_record_v9(
    *,
    execution_sha: str,
    run_uid: str,
    group_ordinal: int,
    task: str,
    content_group_sha256: str,
    sample_key_sha256: str,
    observed_sample_key_sha256: str | None,
    descriptor: Mapping[str, Any],
    projection: Mapping[str, Any],
    raw_prompt_sha256: str,
    canonical_prompt_sha256: str,
    formatter_byte_parity: bool,
    legacy_choice_count: int,
) -> dict[str, Any]:
    """Bind one projection to descriptor, formatter parity and fixed taxonomy."""

    if not isinstance(execution_sha, str) or len(execution_sha) != 40:
        raise ValueError("A5R2 execution SHA must be 40 lowercase hex")
    if any(character not in "0123456789abcdef" for character in execution_sha):
        raise ValueError("A5R2 execution SHA must be 40 lowercase hex")
    if not isinstance(run_uid, str) or not run_uid:
        raise ValueError("A5R2 run UID is empty")
    if isinstance(group_ordinal, bool) or not isinstance(group_ordinal, int) or group_ordinal < 1:
        raise ValueError("A5R2 audit group ordinal must be positive")
    expected_group = _lower_sha256_v9(
        content_group_sha256, name="expected content group"
    )
    _lower_sha256_v9(sample_key_sha256, name="sample key")
    if observed_sample_key_sha256 is None:
        observed_sample_key_sha256 = sample_key_sha256
    _lower_sha256_v9(observed_sample_key_sha256, name="observed sample key")
    raw_prompt_sha256 = _lower_sha256_v9(raw_prompt_sha256, name="raw formatter")
    canonical_prompt_sha256 = _lower_sha256_v9(
        canonical_prompt_sha256, name="canonical formatter"
    )
    if not isinstance(formatter_byte_parity, bool):
        raise ValueError("A5R2 formatter parity must be boolean")
    if formatter_byte_parity != (raw_prompt_sha256 == canonical_prompt_sha256):
        raise ValueError("A5R2 formatter SHA/parity disagree")
    if (
        isinstance(legacy_choice_count, bool)
        or not isinstance(legacy_choice_count, int)
        or legacy_choice_count < 0
    ):
        raise ValueError("A5R2 legacy choice count is invalid")
    expected_descriptor, observed_descriptor = _descriptor_pair_v9(descriptor)
    if (
        projection.get("schema_version") != 9
        or projection.get("protocol_id") != A5R2_PROTOCOL_ID
        or projection.get("artifact_type") != "a5r2_choice_audit_projection"
        or not isinstance(projection.get("persisted"), Mapping)
    ):
        raise ValueError("A5R2 projection header differs")
    persisted = dict(projection["persisted"])
    if "raw_formatter_example" not in projection or "canonical_formatter_example" not in projection:
        raise ValueError("A5R2 projection lacks ephemeral formatter views")
    if persisted.get("expected_descriptor") != expected_descriptor or persisted.get(
        "observed_descriptor"
    ) != observed_descriptor:
        raise ValueError("A5R2 projection/record descriptor differs")
    observed_group = _lower_sha256_v9(
        persisted.get("observed_content_group_sha256"),
        name="observed content group",
    )
    content_match = expected_group == observed_group
    descriptor_match = bool(persisted.get("descriptor_provenance_match"))
    sample_identity_match = (
        sample_key_sha256 == observed_sample_key_sha256
        and all(
            expected_descriptor[field] == observed_descriptor[field]
            for field in A5R2_DESCRIPTOR_IDENTITY_FIELDS
        )
    )
    expected_native = expected_descriptor["non_outcome_native_row_id"]
    observed_native = observed_descriptor["non_outcome_native_row_id"]
    if expected_native is None and observed_native is None:
        native_row_id_status = "NOT_AVAILABLE"
    elif expected_native == observed_native:
        native_row_id_status = "MATCH"
    else:
        native_row_id_status = "MISMATCH"
    identity_status = (
        "MATCH"
        if content_match
        and descriptor_match
        and sample_identity_match
        and native_row_id_status != "MISMATCH"
        else "DESCRIPTOR_INDEX_MISMATCH"
    )
    representation_status = str(persisted.get("representation_status"))
    if representation_status not in {
        "DIRECT",
        "HF_FEATURE_NORMALIZATION_REQUIRED",
        "UNSUPPORTED",
    }:
        raise ValueError("A5R2 representation status differs")
    count = int(persisted["production_choice_count"])
    cardinality_status = _cardinality_status_v9(task, count)
    root_cause, correction_action = _taxonomy_v9(
        task=task,
        count=count,
        identity_status=identity_status,
        representation_status=representation_status,
        cardinality_status=cardinality_status,
        formatter_byte_parity=formatter_byte_parity,
        legacy_choice_count=legacy_choice_count,
    )
    return {
        "schema_version": 9,
        "protocol_id": A5R2_PROTOCOL_ID,
        "artifact_type": "a5r2_choice_cardinality_audit_row",
        "execution_sha": execution_sha,
        "run_uid": run_uid,
        "group_ordinal": group_ordinal,
        "task": task,
        "content_group_sha256": expected_group,
        "sample_key_sha256": sample_key_sha256,
        # Persist the materialization request actually issued by the producer;
        # the completed verifier independently compares it with the frozen
        # split/E0-development descriptor.
        "dataset_id": str(observed_descriptor["dataset_id"]),
        "dataset_config": str(observed_descriptor["dataset_config"]),
        "dataset_split": str(observed_descriptor["dataset_split"]),
        "evaluation_question_id": int(
            observed_descriptor["evaluation_question_id"]
        ),
        "feature_schema_sha256": _lower_sha256_v9(
            observed_descriptor["feature_schema_sha256"],
            name="feature schema",
        ),
        "materialized_container_kind": persisted["choice_container_kind"],
        "legacy_extractor_choice_count": legacy_choice_count,
        "canonical_choice_count": count,
        "source_choice_label_count": len(persisted["source_choice_labels"]),
        "runtime_ordinal_label_count": len(persisted["runtime_ordinal_labels"]),
        "question_nonempty": True,
        "choice_projection_sha256": persisted["choice_projection_sha256"],
        "expected_content_group_sha256": expected_group,
        "observed_content_group_sha256": observed_group,
        "content_identity_match": content_match,
        "sample_identity_match": sample_identity_match,
        "native_row_id_status": native_row_id_status,
        "raw_prompt_sha256": raw_prompt_sha256,
        "canonical_view_prompt_sha256": canonical_prompt_sha256,
        "formatter_byte_parity": formatter_byte_parity,
        "identity_status": identity_status,
        "representation_status": representation_status,
        "cardinality_status": cardinality_status,
        "root_cause_class": root_cause,
        "correction_action": correction_action,
        "gold_or_outcome_field_accessed": False,
    }


def choice_cardinality_audit_row_v9(
    **kwargs: Any,
) -> dict[str, Any]:
    """Compatibility alias for the explicit v9 record builder."""

    return choice_cardinality_record_v9(**kwargs)


def summarize_choice_cardinality_audit_v9(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_task_counts: Mapping[str, int] = EXPECTED_TASK_COUNTS,
) -> dict[str, Any]:
    """Independently reduce the fixed taxonomy to GO or BLOCKED."""

    ordered = [dict(row) for row in rows]
    if [row.get("group_ordinal") for row in ordered] != list(
        range(1, len(ordered) + 1)
    ):
        raise ValueError("A5R2 audit ordinals are not contiguous from one")
    counts = {task: 0 for task in TASK_ORDER}
    by_task_counts = {task: [] for task in TASK_ORDER}
    root_cause_counts = {name: 0 for name in A5R2_ROOT_CAUSE_ACTIONS}
    correction_action_counts = {
        name: 0 for name in set(A5R2_ROOT_CAUSE_ACTIONS.values())
    }
    seen_groups: set[str] = set()
    seen_samples: set[str] = set()
    execution_identity: tuple[str, str] | None = None
    all_content_match = True
    all_sample_identity_match = True
    all_task_cardinalities_valid = True
    all_formatter_equal = True
    zero_forbidden_outcome_access = True
    for row in ordered:
        if (
            row.get("schema_version") != 9
            or row.get("protocol_id") != A5R2_PROTOCOL_ID
            or row.get("artifact_type") != "a5r2_choice_cardinality_audit_row"
            or row.get("gold_or_outcome_field_accessed") is not False
        ):
            raise ValueError("A5R2 audit row header/firewall differs")
        identity = (str(row.get("execution_sha")), str(row.get("run_uid")))
        if execution_identity is None:
            execution_identity = identity
        elif identity != execution_identity:
            raise ValueError("A5R2 audit execution identity changed")
        task = str(row.get("task"))
        if task not in counts:
            raise ValueError("A5R2 audit row has unexpected task")
        group, sample = str(row.get("content_group_sha256")), str(
            row.get("sample_key_sha256")
        )
        if group in seen_groups or sample in seen_samples:
            raise ValueError("A5R2 audit row identity is duplicated")
        seen_groups.add(group)
        seen_samples.add(sample)
        count = int(row["canonical_choice_count"])
        if row.get("runtime_ordinal_label_count") != count:
            raise ValueError("A5R2 audit runtime ordinal count differs")
        if row.get("source_choice_label_count") != count:
            raise ValueError("A5R2 audit source label count differs")
        root_cause = str(row.get("root_cause_class"))
        action = str(row.get("correction_action"))
        if root_cause not in A5R2_ROOT_CAUSE_ACTIONS or action != A5R2_ROOT_CAUSE_ACTIONS[
            root_cause
        ]:
            raise ValueError("A5R2 audit root-cause/action mapping differs")
        if row.get("identity_status") not in {"MATCH", "DESCRIPTOR_INDEX_MISMATCH"}:
            raise ValueError("A5R2 audit identity status differs")
        if row.get("representation_status") not in {
            "DIRECT",
            "HF_FEATURE_NORMALIZATION_REQUIRED",
            "UNSUPPORTED",
        }:
            raise ValueError("A5R2 audit representation status differs")
        if row.get("cardinality_status") not in {"TASK_VALID", "TASK_INVALID"}:
            raise ValueError("A5R2 audit cardinality status differs")
        all_content_match &= row.get("content_identity_match") is True
        all_sample_identity_match &= (
            row.get("sample_identity_match") is True
            and row.get("native_row_id_status") != "MISMATCH"
        )
        all_task_cardinalities_valid &= row.get("cardinality_status") == "TASK_VALID"
        all_formatter_equal &= row.get("formatter_byte_parity") is True
        zero_forbidden_outcome_access &= (
            row.get("gold_or_outcome_field_accessed") is False
        )
        counts[task] += 1
        by_task_counts[task].append(count)
        root_cause_counts[root_cause] += 1
        correction_action_counts[action] += 1
    counts_match = counts == dict(expected_task_counts)
    required_zero = all(
        root_cause_counts[name] == 0 for name in A5R2_REQUIRED_ZERO_ROOT_CAUSES
    )
    status = (
        "A5R2_CHOICE_AUDIT_GO"
        if counts_match
        and required_zero
        and all_content_match
        and all_sample_identity_match
        and all_task_cardinalities_valid
        and all_formatter_equal
        and zero_forbidden_outcome_access
        else "A5R2_CHOICE_AUDIT_BLOCKED"
    )
    return {
        "schema_version": 9,
        "protocol_id": A5R2_PROTOCOL_ID,
        "artifact_type": "a5r2_choice_cardinality_audit_summary",
        "status": status,
        "execution_sha": execution_identity[0] if execution_identity else None,
        "run_uid": execution_identity[1] if execution_identity else None,
        "row_count": len(ordered),
        "task_counts": counts,
        "expected_task_counts": dict(expected_task_counts),
        "task_counts_match": counts_match,
        "production_choice_count_distribution": _count_distribution(
            [int(row["canonical_choice_count"]) for row in ordered]
        ),
        "by_task_production_choice_count_distribution": {
            task: _count_distribution(values) for task, values in by_task_counts.items()
        },
        "root_cause_counts": root_cause_counts,
        "correction_action_counts": correction_action_counts,
        "required_zero_root_causes_satisfied": required_zero,
        "all_content_group_hashes_match": all_content_match,
        "all_sample_group_split_identities_unchanged": all_sample_identity_match,
        "all_task_cardinalities_valid": all_task_cardinalities_valid,
        "all_rows_formatter_byte_equal": all_formatter_equal,
        "zero_forbidden_outcome_access": zero_forbidden_outcome_access,
        "semantic_sha256": sha256_bytes(canonical_json_bytes(ordered)),
    }


def build_choice_cardinality_audit_lock_v9(
    rows: Sequence[Mapping[str, Any]],
    *,
    execution_sha: str,
    run_uid: str,
    run_root: str,
    ledger_artifact: Mapping[str, Any],
    summary_artifact: Mapping[str, Any],
    e0_data_tree_binding: Mapping[str, Any],
    source_snapshot_binding: Mapping[str, Any],
    source_bindings: Sequence[Mapping[str, Any]],
    predecessor_v8_unchanged: bool,
    independent_reduction_and_hash_verified: bool,
    frozen_group_sample_membership_verified: bool,
    frozen_canonical_order_verified: bool,
    frozen_loader_coordinates_verified: bool,
    expected_task_counts: Mapping[str, int] = EXPECTED_TASK_COUNTS,
) -> dict[str, Any]:
    summary = summarize_choice_cardinality_audit_v9(
        rows, expected_task_counts=expected_task_counts
    )
    if summary["execution_sha"] != execution_sha or summary["run_uid"] != run_uid:
        raise ValueError("A5R2 audit lock execution identity differs")

    def artifact_descriptor(
        value: Mapping[str, Any], *, name: str, row_count: int
    ) -> dict[str, Any]:
        artifact = dict(value)
        if set(artifact) != {"relative_path", "sha256", "bytes", "row_count"}:
            raise ValueError(f"A5R2 {name} descriptor fields differ")
        if not isinstance(artifact["relative_path"], str) or not artifact[
            "relative_path"
        ]:
            raise ValueError(f"A5R2 {name} relative path is empty")
        _lower_sha256_v9(artifact["sha256"], name=name)
        if (
            isinstance(artifact["bytes"], bool)
            or not isinstance(artifact["bytes"], int)
            or artifact["bytes"] < 0
            or artifact["row_count"] != row_count
        ):
            raise ValueError(f"A5R2 {name} size/row count differs")
        return artifact

    artifact = artifact_descriptor(
        ledger_artifact, name="audit ledger", row_count=len(rows)
    )
    summary_descriptor = artifact_descriptor(
        summary_artifact, name="audit summary", row_count=1
    )
    # Persisted JSON artifacts in the input-lock producer are canonical JSON
    # terminated by exactly one LF.  Bind those exact bytes, not merely the
    # parsed object or the newline-free pure-function representation.
    summary_bytes = canonical_json_bytes(summary) + b"\n"
    if (
        summary_descriptor["sha256"] != sha256_bytes(summary_bytes)
        or summary_descriptor["bytes"] != len(summary_bytes)
    ):
        raise ValueError("A5R2 audit summary descriptor does not bind the summary")

    expected_e0_binding = {
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
    e0_binding = dict(e0_data_tree_binding)
    if e0_binding != expected_e0_binding:
        raise ValueError("A5R2 pre-audit E0 dual-domain binding differs")

    snapshot_binding = dict(source_snapshot_binding)
    snapshot_fields = {
        "receipt_path",
        "receipt_file_sha256",
        "receipt_sha256",
        "mounted_tree_sha256",
        "execution_sha",
        "independently_verified_before_audit_row_one",
    }
    if set(snapshot_binding) != snapshot_fields:
        raise ValueError("A5R2 source-snapshot binding fields differ")
    if not isinstance(snapshot_binding["receipt_path"], str) or not snapshot_binding[
        "receipt_path"
    ]:
        raise ValueError("A5R2 source-snapshot receipt path is empty")
    for field in (
        "receipt_file_sha256",
        "receipt_sha256",
        "mounted_tree_sha256",
    ):
        _lower_sha256_v9(snapshot_binding[field], name=f"source snapshot {field}")
    if (
        snapshot_binding["execution_sha"] != execution_sha
        or snapshot_binding["independently_verified_before_audit_row_one"]
        is not True
    ):
        raise ValueError("A5R2 source snapshot was not verified before row one")
    bindings = [dict(binding) for binding in source_bindings]
    if not bindings or any(set(binding) != {"path", "sha256"} for binding in bindings):
        raise ValueError("A5R2 source bindings are empty or malformed")
    for binding in bindings:
        if not isinstance(binding["path"], str) or not binding["path"]:
            raise ValueError("A5R2 source binding path is empty")
        _lower_sha256_v9(binding["sha256"], name="source binding")
    if len({canonical_json_bytes(binding) for binding in bindings}) != len(bindings):
        raise ValueError("A5R2 source bindings contain duplicates")
    taxonomy_counts = dict(summary["root_cause_counts"])
    cardinality_by_task: dict[str, dict[str, int]] = {}
    for task in TASK_ORDER:
        bins = {name: 0 for name in ("zero", "one", "two", "three", "four", "five_plus")}
        for row in rows:
            if row["task"] != task:
                continue
            count = int(row["canonical_choice_count"])
            key = (
                "zero" if count == 0 else
                "one" if count == 1 else
                "two" if count == 2 else
                "three" if count == 3 else
                "four" if count == 4 else
                "five_plus"
            )
            bins[key] += 1
        cardinality_by_task[task] = bins
    hard_checks = {
        "e0_data_tree_dual_domains_verified_before_audit": True,
        "source_snapshot_receipt_and_tree_verified_before_audit": True,
        "exact_population_and_task_counts": summary["task_counts_match"] and len(rows) == 326,
        "unique_group_and_sample_identity": bool(
            frozen_group_sample_membership_verified
        ),
        "canonical_order_from_group_one": bool(
            frozen_canonical_order_verified
        ),
        "all_content_group_hashes_match": summary["all_content_group_hashes_match"],
        "all_sample_group_split_identities_unchanged": (
            summary["all_sample_group_split_identities_unchanged"]
            and bool(frozen_loader_coordinates_verified)
        ),
        "all_task_cardinalities_valid": summary["all_task_cardinalities_valid"],
        "all_rows_formatter_byte_equal": summary[
            "all_rows_formatter_byte_equal"
        ],
        "no_pseudo_reorder_truncation_replacement_or_remap": all(
            row["root_cause_class"]
            not in {
                "DESCRIPTOR_INDEX_MISMATCH",
                "UNSUPPORTED_FEATURE_REPRESENTATION",
                "UNEXPLAINED_OTHER",
            }
            for row in rows
        ),
        "zero_forbidden_outcome_access": summary["zero_forbidden_outcome_access"],
        "predecessor_v8_unchanged": bool(predecessor_v8_unchanged),
        "independent_reduction_and_hash_verified": bool(
            independent_reduction_and_hash_verified
        ),
    }
    go = summary["status"] == "A5R2_CHOICE_AUDIT_GO" and all(hard_checks.values())
    status = "A5R2_CHOICE_AUDIT_GO" if go else "A5R2_CHOICE_AUDIT_BLOCKED"
    selected_actions = (
        sorted(
            {
                str(row["correction_action"])
                for row in rows
                if row["correction_action"] != "BLOCK"
            }
        )
        if go
        else []
    )
    payload = {
        "schema_version": 9,
        "protocol_id": A5R2_PROTOCOL_ID,
        "artifact_type": "a5r2_choice_cardinality_audit_lock",
        "execution_sha": execution_sha,
        "run_uid": run_uid,
        "run_root": run_root,
        "status": status,
        "population_count": len(rows),
        "task_counts": dict(summary["task_counts"]),
        "taxonomy_counts": taxonomy_counts,
        "cardinality_by_task": cardinality_by_task,
        "ledger": artifact,
        "summary": summary_descriptor,
        "e0_data_tree_binding": e0_binding,
        "source_snapshot_binding": snapshot_binding,
        "source_bindings": bindings,
        "hard_checks": hard_checks,
        "correction_actions_selected": selected_actions,
        "mechanical_decision": status,
        "audit_semantic_sha256": summary["semantic_sha256"],
        "resume_allowed": False,
        "same_execution_input_lock_consumption_allowed": go,
        "model_or_checkpoint_loaded": False,
        "model_forward_run": False,
        "gpu_cuda_or_kubernetes_used": False,
        "training": False,
        "e1_2_or_e1_3_authorized": False,
        "e1_pilot_or_confirmatory_accessed": False,
    }
    payload["evidence_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


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


def _dual_anchor_record_v9(
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
    """Build the explicit v9 record without changing the immutable v7/v8 path."""

    production_payload = choice_payload_v9(task, example)
    projected = historical_projected_example_v9(task, example)
    historical_payload = choice_payload_v9(task, projected)
    recomputed_historical_content = canonical_historical_content_sha256(
        str(historical_payload["question"]),
        list(historical_payload["production_choices"]),
    )
    if not (
        recomputed_historical_content
        == historical_content_sha256
        == content_group_sha256
    ):
        raise ValueError("A5R2 historical content-group anchor changed")
    relation, choice_only = classify_prompt_relation_v9(
        historical_payload=historical_payload,
        production_payload=production_payload,
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
        raise ValueError("A5R2 census contains an invalid nonnegative count")
    historical_identity = {
        "question": historical_payload["question"],
        "choices": historical_payload["production_choices"],
    }
    label_free_runtime_identity = {
        "task": task,
        "question": production_payload["question"],
        "production_choices": production_payload["production_choices"],
        "source_choice_labels": production_payload["source_choice_labels"],
        "runtime_ordinal_labels": production_payload["runtime_ordinal_labels"],
    }
    return {
        "schema_version": 9,
        "protocol_id": A5R2_PROTOCOL_ID,
        "artifact_type": "a5_prompt_census_record",
        "task": task,
        "content_group_sha256": content_group_sha256,
        "sample_key_sha256": sample_key_sha256,
        "source_row_id": str(source_row_id),
        "historical_choice_count": int(
            historical_payload["production_choice_count"]
        ),
        "production_choice_count": int(
            production_payload["production_choice_count"]
        ),
        "source_choice_labels": list(production_payload["source_choice_labels"]),
        "runtime_ordinal_labels": list(
            production_payload["runtime_ordinal_labels"]
        ),
        "gold_answer": gold_answer,
        "historical_first4_question_choices_sha256": sha256_bytes(
            canonical_json_bytes(historical_identity)
        ),
        "historical_rendered_prompt_sha256": sha256_bytes(
            historical_rendered_prompt.encode("utf-8")
        ),
        "historical_alignment_sha256": historical_alignment_sha256,
        "label_free_runtime_row_sha256": sha256_bytes(
            canonical_json_bytes(label_free_runtime_identity)
        ),
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
        "choice_audit_forbidden_outcome_field_access_count": 0,
    }


def dual_anchor_record_v9(**kwargs: Any) -> dict[str, Any]:
    """Public explicit-v9 constructor used by the corrected input lock."""

    return _dual_anchor_record_v9(**kwargs)


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
    schema_version: int = 7,
    protocol_id: str = A5_PROTOCOL_ID,
) -> dict[str, Any]:
    schema_version, protocol_id = _validated_protocol_header(
        schema_version, protocol_id
    )
    if schema_version == 9:
        return _dual_anchor_record_v9(
            task=task,
            content_group_sha256=content_group_sha256,
            sample_key_sha256=sample_key_sha256,
            source_row_id=source_row_id,
            example=example,
            gold_answer=gold_answer,
            historical_rendered_prompt=historical_rendered_prompt,
            historical_alignment_sha256=historical_alignment_sha256,
            production_rendered_prompt=production_rendered_prompt,
            production_alignment_sha256=production_alignment_sha256,
            historical_prompt_token_count=historical_prompt_token_count,
            production_prompt_token_count=production_prompt_token_count,
            production_certified_parent_count=production_certified_parent_count,
            production_logical_row_count=production_logical_row_count,
            production_physical_chunk_count=production_physical_chunk_count,
            historical_content_sha256=historical_content_sha256,
        )
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
        "schema_version": schema_version,
        "protocol_id": protocol_id,
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
    schema_version: int = 7,
    protocol_id: str = A5_PROTOCOL_ID,
    choice_audit_lock_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail-closed full-population reduction in canonical task/group order."""

    schema_version, protocol_id = _validated_protocol_header(
        schema_version, protocol_id
    )
    if schema_version == 9:
        choice_audit_lock_sha256 = _lower_sha256_v9(
            choice_audit_lock_sha256, name="choice-audit lock"
        )
    elif choice_audit_lock_sha256 is not None:
        raise ValueError("legacy A5 census may not bind an A5R2 audit lock")

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
        if (
            row.get("schema_version") != schema_version
            or row.get("protocol_id") != protocol_id
        ):
            raise ValueError("A5 census row protocol header is inconsistent")
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
        if schema_version == 9:
            production_count = row.get("production_choice_count")
            historical_count = row.get("historical_choice_count")
            source_labels = row.get("source_choice_labels")
            runtime_labels = row.get("runtime_ordinal_labels")
            if (
                not isinstance(production_count, int)
                or isinstance(production_count, bool)
                or not isinstance(historical_count, int)
                or isinstance(historical_count, bool)
                or historical_count != min(4, production_count)
                or not isinstance(source_labels, list)
                or len(source_labels) != production_count
                or len(set(source_labels)) != len(source_labels)
                or runtime_labels
                != [chr(65 + index) for index in range(production_count)]
            ):
                raise ValueError("A5R2 census choice geometry is invalid")
            if task == "ai2-arc" and production_count < 2:
                raise ValueError("A5R2 census ARC choice count is below two")
            if task in {"openbookqa", "mmlu-redux"} and production_count != 4:
                raise ValueError("A5R2 census fixed-cardinality task changed")
            validate_gold_answer_v9(
                str(row.get("gold_answer")),
                source_choice_labels=source_labels,
                runtime_ordinal_labels=runtime_labels,
            )
            if row.get("choice_audit_forbidden_outcome_field_access_count") != 0:
                raise ValueError("A5R2 census lacks the label-free audit firewall")
            _lower_sha256_v9(
                row.get("choice_audit_row_sha256"),
                name="census choice-audit row",
            )
            if row.get("choice_audit_lock_sha256") != choice_audit_lock_sha256:
                raise ValueError("A5R2 census choice-audit lock binding changed")
            if row.get("choice_root_cause_class") not in A5R2_ROOT_CAUSE_ACTIONS:
                raise ValueError("A5R2 census choice root-cause class changed")
        else:
            if row.get("gold_answer") not in GOLD_ANSWERS:
                raise ValueError("A5 census contains a gold answer outside A-D")
            if row.get("historical_choice_count") != 4:
                raise ValueError("A5 census historical choice count changed")
            labels = row.get("raw_choice_labels")
            production_count = row.get("production_choice_count")
            if (
                not isinstance(labels, list)
                or not isinstance(production_count, int)
                or labels
                != [chr(65 + index) for index in range(production_count)]
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
    result = {
        "schema_version": schema_version,
        "protocol_id": protocol_id,
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
    if schema_version == 9:
        result["choice_audit_lock_sha256"] = choice_audit_lock_sha256
    return result


def build_census_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    execution_sha: str,
    run_uid: str,
    record_artifact: Mapping[str, Any],
    expected_task_counts: Mapping[str, int] = EXPECTED_TASK_COUNTS,
    schema_version: int = 7,
    protocol_id: str = A5_PROTOCOL_ID,
    choice_audit_lock_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the strict A5 schema manifest after semantic census validation."""

    summary = summarize_dual_anchor_census(
        records,
        expected_task_counts=expected_task_counts,
        schema_version=schema_version,
        protocol_id=protocol_id,
        choice_audit_lock_sha256=choice_audit_lock_sha256,
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
        (
            "label_free_runtime_row_sha256"
            if schema_version == 9
            else "raw_full_row_sha256"
        ),
        "production_rendered_prompt_sha256",
        "production_alignment_sha256",
        "prompt_relation",
    )
    if schema_version == 9:
        anchor_fields = (
            *anchor_fields,
            "choice_audit_row_sha256",
            "choice_root_cause_class",
        )
    artifact = dict(record_artifact)
    expected_artifact_fields = {"relative_path", "sha256", "bytes", "row_count"}
    if set(artifact) != expected_artifact_fields or artifact["row_count"] != len(
        ordered
    ):
        raise ValueError("A5 census record artifact descriptor is invalid")
    result = {
        "schema_version": schema_version,
        "protocol_id": protocol_id,
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
    if schema_version == 9:
        result["choice_audit_lock_sha256"] = choice_audit_lock_sha256
    return result


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
