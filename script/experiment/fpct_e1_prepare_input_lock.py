#!/usr/bin/env python3
"""CPU-only canonical-input locker for FPCT-E1 E0-design mechanism shards.

The command tokenizes and aligns only the already-open E0-design population. It
loads no model class, checkpoint or CUDA device.  Its local PT sidecar contains
the exact full-response ``AlignedChatDataset`` feature, prompt-only production
generation inputs, certified runtime parents and answer-query identities.  The
compact JSON manifest freezes hashes/counts and the model-independent row-key
template universe before any E1 model output is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from script.experiment.fpct_e1_capture_runner import (
    E0_DEV_MANIFEST_RELATIVE,
    TASKS,
    TASK_GROUP_COUNTS,
    load_e0_design_lock,
    sha256_file,
    verify_e0_dev_anchor,
)
from script.experiment.fpct_e1_runtime_backend import (
    GOLD_RESPONSE_TEMPLATE,
    _alignment_sha256,
    _build_aligner,
    _instruction_end,
    _load_task_example,
    _prompt_alignment_details,
    _question_choices,
    _sha256_bytes,
    _to_python,
    _verify_feature_alignment,
    canonical_content_sha256,
    canonical_sample_sha256,
    feature_provenance,
    topology_contract,
)


SCHEMA_VERSION = 1
PROTOCOL_ID = "fpct_e1_e0_design_input_lock_v1"
EXPECTED_RECEIVER_LAYERS = 28
EXPECTED_QUERY_HEADS = 16
MAX_LONG_FORM_ROWS_PER_SAMPLE = 262144


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def nested_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(_to_python(value))).hexdigest()


def task_membership_sha256(
    group_contract: Sequence[Mapping[str, Any]],
    semantic_members: Sequence[Mapping[str, Any]],
) -> str:
    """Mechanically bind plan membership to the sidecar's semantic item SHAs."""

    planned: dict[str, str] = {}
    for row in group_contract:
        group = str(row["content_group_sha256"])
        samples = row.get("sample_sha256")
        if not isinstance(samples, list) or len(samples) != 1:
            raise ValueError("input-lock membership requires one sample per group")
        if group in planned:
            raise ValueError("duplicate group in input-lock plan membership")
        planned[group] = str(samples[0])
    observed: dict[str, tuple[str, str]] = {}
    for row in semantic_members:
        group = str(row["content_group_sha256"])
        value = (str(row["sample_sha256"]), str(row["item_semantic_sha256"]))
        if group in observed:
            raise ValueError("duplicate group in input-lock semantic membership")
        if any(len(item) != 64 for item in (group, *value)):
            raise ValueError("input-lock membership contains a malformed SHA")
        observed[group] = value
    if set(planned) != set(observed):
        raise ValueError("plan and semantic input-lock group universes differ")
    if any(planned[group] != observed[group][0] for group in planned):
        raise ValueError("plan and semantic input-lock sample identities differ")
    rows = sorted(
        [sample, group, observed[group][1]] for group, sample in planned.items()
    )
    return nested_sha256(rows)


def answer_query_contract(labels: Sequence[Any]) -> list[dict[str, int]]:
    result = []
    values = [int(value) for value in labels]
    for query in range(max(0, len(values) - 1)):
        token = values[query + 1]
        if token != -100:
            if token < 0:
                raise ValueError("eligible answer target token is negative")
            result.append(
                {
                    "query_position": query,
                    "target_position": query + 1,
                    "target_token_id": token,
                }
            )
    if not result:
        raise ValueError("canonical response contains no eligible answer query")
    return result


def certified_parent_contract(
    topology: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    parents = []
    for parent, record in sorted(topology.items()):
        count = int(record["candidate_count"])
        if count < 2:
            continue
        if not record["within_instruction"]:
            raise ValueError("response parent entered include_response=false input lock")
        parents.append(
            {
                "parent_position": int(parent),
                "candidate_count": count,
                "prior": [float(value) for value in record["prior"]],
                "topology": record["topology"],
            }
        )
    if not parents:
        raise ValueError("certified E0-design sample contains no runtime m>=2 parent")
    return parents


BOUNDARY_REASONS = {
    "zero_length_receiver_interval",
    "duplicate_or_overlap_receiver_offsets",
    "zero_length_source_interval",
    "exact_duplicate_source_offsets",
    "partial_overlap_source_offsets",
    "candidate_without_receiver_intersection",
}
ALIAS_REASONS = {
    "exact_duplicate_source_offsets",
    "partial_overlap_source_offsets",
    "duplicate_or_overlap_receiver_offsets",
}


def _relative_interval(
    offset: Sequence[Any], content_span: Sequence[Any]
) -> tuple[int, int]:
    """Clip one absolute tokenizer offset to message-relative content geometry."""

    start, end = (int(value) for value in offset)
    content_start, content_end = (int(value) for value in content_span)
    return (
        max(0, min(content_end, start) - content_start),
        max(0, min(content_end, end) - content_start),
    )


def _intersection(
    receiver: tuple[int, int], source: tuple[int, int]
) -> tuple[int, int] | None:
    left, right = max(receiver[0], source[0]), min(receiver[1], source[1])
    return (left, right) if right > left else None


def raw_topology_ledger(
    *,
    raw_details: Mapping[str, Any],
    sanitized_details: Mapping[str, Any],
    instruction_end: int,
    task: str,
    sample_sha256: str,
    content_group_sha256: str,
    candidate_window: int,
) -> list[dict[str, Any]]:
    """Conservatively classify every pre-sanitizer prompt parent with raw m>=2."""

    from script.analysis.fpct_e1_mechanism_audit import (
        classify_candidate_topology,
    )

    candidate_window = int(candidate_window)
    if candidate_window < 0:
        raise ValueError("candidate_window must be nonnegative")
    raw = raw_details["soft_alignment"]
    runtime = sanitized_details["soft_alignment"]
    message_sections = [
        section for section in raw_details.get("sections", [])
        if section.get("type") == "message"
    ]
    content_spans_slm = raw_details.get("content_spans_slm", [])
    content_spans_llm = raw_details.get("content_spans_llm", [])
    if not (
        len(message_sections)
        == len(content_spans_slm)
        == len(content_spans_llm)
    ):
        raise ValueError("raw topology requires matched message/content spans")
    rows = []
    for parent in range(min(instruction_end, len(raw["source_indices"]))):
        if not bool(raw_details["message_mask"][parent]):
            continue
        raw_slots = [
            (slot, int(index), float(weight))
            for slot, (index, weight) in enumerate(
                zip(raw["source_indices"][parent], raw["source_weights"][parent])
            )
            if int(index) >= 0 and float(weight) > 0
        ]
        raw_legal = [index for _slot, index, _weight in raw_slots]
        if len(raw_legal) < 2:
            continue
        if len(raw_legal) > 4:
            raise ValueError("raw topology exceeds frozen top-k=4")
        if len(set(raw_legal)) != len(raw_legal):
            raise ValueError("raw topology contains duplicate source indices")
        raw_weights = [weight for _slot, _index, weight in raw_slots]
        if any(not math.isfinite(weight) for weight in raw_weights) or not math.isclose(
            sum(raw_weights), 1.0, abs_tol=2e-5, rel_tol=0
        ):
            raise ValueError("raw topology prior is nonfinite or not normalized")
        runtime_slots = [
            (slot, int(index), float(weight))
            for slot, (index, weight) in enumerate(
                zip(
                    runtime["source_indices"][parent],
                    runtime["source_weights"][parent],
                )
            )
            if int(index) >= 0 and float(weight) > 0
        ]
        runtime_legal = [index for _slot, index, _weight in runtime_slots]
        runtime_weights = [weight for _slot, _index, weight in runtime_slots]
        if not 1 <= len(runtime_legal) <= 4:
            raise ValueError("raw topology runtime support must satisfy 1<=m<=4")
        if len(set(runtime_legal)) != len(runtime_legal):
            raise ValueError("raw topology runtime contains duplicate source indices")
        if any(
            not math.isfinite(weight) for weight in runtime_weights
        ) or not math.isclose(sum(runtime_weights), 1.0, abs_tol=2e-5, rel_tol=0):
            raise ValueError("raw topology runtime prior is nonfinite or not normalized")
        runtime_weight_by_index = {
            index: weight for _slot, index, weight in runtime_slots
        }
        certified = bool(runtime["fpct_certified_mask"][parent])
        reason = str(runtime["fpct_certification_reason"][parent])
        section_matches = [
            (ordinal, section)
            for ordinal, section in enumerate(message_sections)
            if int(section["slm_range"][0]) <= parent < int(section["slm_range"][1])
        ]
        if len(section_matches) != 1:
            raise ValueError("raw topology parent has no unique message section")
        ordinal, section = section_matches[0]
        receiver_offset = tuple(
            int(value) for value in raw_details["slm_offsets"][parent]
        )
        receiver_span = _relative_interval(
            receiver_offset, content_spans_slm[ordinal]
        )
        candidate_records = []
        for slot, index, raw_weight in raw_slots:
            if index >= len(raw_details["llm_offsets"]) or index >= len(
                raw_details["llm_ids"]
            ):
                raise ValueError("raw topology candidate index is out of range")
            source_offset = tuple(
                int(value) for value in raw_details["llm_offsets"][index]
            )
            in_message = (
                int(section["llm_range"][0])
                <= index
                < int(section["llm_range"][1])
            )
            source_span = _relative_interval(
                source_offset, content_spans_llm[ordinal]
            )
            intersection = _intersection(receiver_span, source_span)
            if intersection is not None and in_message:
                origin = "span_overlap"
            elif candidate_window > 0 and in_message:
                origin = "window_neighbor"
            else:
                origin = "fallback_or_unknown"
            complete = (
                intersection is not None
                and intersection[0] == receiver_span[0]
                and intersection[1] == receiver_span[1]
            )
            candidate_records.append(
                {
                    "slot": slot,
                    "source_index": index,
                    "source_token_id": int(raw_details["llm_ids"][index]),
                    "source_offset": list(source_offset),
                    "source_span": list(source_span),
                    "intersection": (
                        list(intersection) if intersection is not None else None
                    ),
                    "intersection_length": (
                        intersection[1] - intersection[0]
                        if intersection is not None
                        else 0
                    ),
                    "origin": origin,
                    "raw_weight": raw_weight,
                    "runtime_retained": index in runtime_weight_by_index,
                    "runtime_weight": runtime_weight_by_index.get(index, 0.0),
                    "complete_receiver_explanation": complete,
                }
            )
        source_spans = [
            tuple(record["source_span"]) for record in candidate_records
        ]
        duplicate_offsets = len(set(source_spans)) != len(source_spans)
        alias = reason in ALIAS_REASONS or duplicate_offsets
        boundary = (
            reason in BOUNDARY_REASONS
            or receiver_span[1] <= receiver_span[0]
            or any(record["origin"] == "fallback_or_unknown" for record in candidate_records)
        )
        all_complete = all(
            record["complete_receiver_explanation"] for record in candidate_records
        )
        independent_complete = (
            all_complete
            and len({record["source_token_id"] for record in candidate_records})
            == len(candidate_records)
            and not duplicate_offsets
        )
        taxonomy = classify_candidate_topology(
            receiver_span,
            source_spans,
            candidate_origins=[record["origin"] for record in candidate_records],
            boundary_or_fallback=boundary,
            independent_competitors=independent_complete,
            certified_partition=certified,
            duplicate_or_overlap_alias=alias,
        )
        if certified:
            if (
                len(runtime_legal) < 2
                or reason != "certified_disjoint_partition"
                or taxonomy != "partition_compositional"
            ):
                raise ValueError("certified raw topology row lost its runtime partition")
        elif len(runtime_legal) > 1:
            raise ValueError("uncertified raw topology row was not slot-0 collapsed")
        span_payload = {
            "receiver_span": receiver_span,
            "candidates": candidate_records,
        }
        rows.append(
            {
                "schema_version": 1,
                "split_role": "e0_design",
                "task": task,
                "sample_sha256": sample_sha256,
                "content_group_sha256": content_group_sha256,
                "parent_position": parent,
                "receiver_token_id": int(raw_details["slm_ids"][parent]),
                "receiver_offset": list(receiver_offset),
                "receiver_span": list(receiver_span),
                "raw_candidate_count": len(raw_legal),
                "runtime_candidate_count": len(runtime_legal),
                "raw_candidate_indices": raw_legal,
                "runtime_candidate_indices": runtime_legal,
                "raw_weights": raw_weights,
                "runtime_weights": runtime_weights,
                "candidates": candidate_records,
                "certified": certified,
                "offset_uncertified": not certified,
                "certification_reason": reason,
                "taxonomy": taxonomy,
                "candidate_window": candidate_window,
                "duplicate_or_overlap_alias": alias,
                "runtime_functional_eligible": bool(
                    certified and len(runtime_legal) >= 2
                ),
                "functional_metrics_present": False,
                "span_geometry_sha256": nested_sha256(span_payload),
            }
        )
    return rows


def row_template_records(
    item: Mapping[str, Any],
    *,
    num_layers: int,
    num_query_heads: int,
    num_kv_heads: int,
) -> list[list[Any]]:
    """Return sorted model-independent suffixes of the executor row key."""

    if num_query_heads % num_kv_heads:
        raise ValueError("receiver Hq is not divisible by Hkv")
    group = num_query_heads // num_kv_heads
    rows: list[list[Any]] = []
    for layer in range(num_layers):
        for query_head in range(num_query_heads):
            kv_head = query_head // group
            for query in item["answer_queries"]:
                for parent in item["certified_parents"]:
                    rows.append(
                        [
                            item["sample_sha256"],
                            item["content_group_sha256"],
                            item["provenance"]["input_sha256"],
                            item["provenance"]["alignment_sha256"],
                            item["provenance"]["labels_sha256"],
                            item["provenance"]["gold_response_sha256"],
                            layer,
                            query_head,
                            kv_head,
                            query["query_position"],
                            query["target_position"],
                            query["target_token_id"],
                            parent["parent_position"],
                            parent["candidate_count"],
                            parent["topology"],
                        ]
                    )
    return sorted(rows, key=canonical_json_bytes)


def expected_long_form_rows(
    *,
    answer_query_count: int,
    certified_parent_count: int,
    num_layers: int,
    num_query_heads: int,
) -> int:
    values = (
        answer_query_count,
        certified_parent_count,
        num_layers,
        num_query_heads,
    )
    if any(isinstance(value, bool) or int(value) != value or value <= 0 for value in values):
        raise ValueError("long-form row factors must be positive integers")
    count = int(answer_query_count * certified_parent_count * num_layers * num_query_heads)
    if count > MAX_LONG_FORM_ROWS_PER_SAMPLE:
        raise ValueError(
            f"input-lock sample exceeds long-form row ceiling: {count} > {MAX_LONG_FORM_ROWS_PER_SAMPLE}"
        )
    return count


def expected_long_form_row_volume(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze exact task volume; quantiles use sorted[ceil(p*n)-1]."""

    if not items:
        raise ValueError("cannot freeze an empty task row-volume contract")
    ordered = sorted(int(item["expected_long_form_rows"]) for item in items)
    if any(value <= 0 or value > MAX_LONG_FORM_ROWS_PER_SAMPLE for value in ordered):
        raise ValueError("task row-volume contains an invalid per-sample count")
    count = len(ordered)

    def nearest_rank(probability: float) -> int:
        return ordered[math.ceil(probability * count) - 1]

    maximum = ordered[-1]
    argmax = min(
        (item for item in items if int(item["expected_long_form_rows"]) == maximum),
        key=lambda item: (item["sample_sha256"], item["content_group_sha256"]),
    )
    return {
        "formula": "num_layers*num_query_heads*answer_query_count*certified_parent_count",
        "sample_ceiling": MAX_LONG_FORM_ROWS_PER_SAMPLE,
        "count": count,
        "sum": sum(ordered),
        "min": ordered[0],
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
        "max": maximum,
        "argmax": {
            "sample_sha256": argmax["sample_sha256"],
            "content_group_sha256": argmax["content_group_sha256"],
        },
        "quantile_method": "nearest_rank",
        "quantile_definition": "sorted[ceil(p*n)-1]",
    }


def row_template_attestation(
    items: Sequence[Mapping[str, Any]],
    *,
    num_layers: int,
    num_query_heads: int,
    num_kv_heads: int,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    previous: bytes | None = None
    for item in sorted(
        items, key=lambda value: (value["sample_sha256"], value["content_group_sha256"])
    ):
        for row in row_template_records(
            item,
            num_layers=num_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
        ):
            payload = canonical_json_bytes(row).rstrip(b"\n")
            if previous is not None and payload <= previous:
                raise ValueError("input lock contains duplicate/nonmonotonic row template")
            previous = payload
            digest.update(payload + b"\n")
            count += 1
    if count == 0:
        raise ValueError("input lock row template universe is empty")
    return {"count": count, "sha256": digest.hexdigest()}


def _config_dimensions(receiver_path: Path) -> dict[str, int]:
    config_path = receiver_path / "config.json"
    value = json.loads(config_path.read_text(encoding="utf-8"))
    dimensions = {
        "num_hidden_layers": int(value["num_hidden_layers"]),
        "num_attention_heads": int(value["num_attention_heads"]),
        "num_key_value_heads": int(value["num_key_value_heads"]),
    }
    if dimensions["num_hidden_layers"] != EXPECTED_RECEIVER_LAYERS:
        raise ValueError("receiver layer count differs from frozen 28")
    if dimensions["num_attention_heads"] != EXPECTED_QUERY_HEADS:
        raise ValueError("receiver query-head count differs from frozen 16")
    if dimensions["num_attention_heads"] % dimensions["num_key_value_heads"]:
        raise ValueError("receiver GQA head mapping is invalid")
    return dimensions


def _tokenizer_files(path: Path) -> list[dict[str, Any]]:
    names = (
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    )
    records = []
    for name in names:
        candidate = path / name
        if candidate.is_file():
            records.append(
                {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)}
            )
    if not records:
        raise FileNotFoundError(f"no tokenizer/config files at {path}")
    return records


def runtime_asset_tree(path: Path) -> dict[str, Any]:
    """Freeze every file in one resolved pretrained runtime asset tree."""

    requested = path.absolute()
    if not requested.exists():
        raise FileNotFoundError(requested)
    root_is_symlink = requested.is_symlink()
    root_link_target = os.readlink(requested) if root_is_symlink else None
    resolved = requested.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("runtime asset root is not a directory")
    files: list[dict[str, Any]] = []
    for candidate in sorted(resolved.rglob("*")):
        relative = candidate.relative_to(resolved).as_posix()
        if candidate.is_symlink():
            target_text = os.readlink(candidate)
            target = candidate.resolve(strict=True)
            if target.is_dir():
                raise ValueError(
                    f"runtime asset tree contains unsupported directory symlink: {relative}"
                )
            files.append(
                {
                    "relative_path": relative,
                    "kind": "symlink_file",
                    "symlink_target": target_text,
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
            )
        elif candidate.is_file():
            files.append(
                {
                    "relative_path": relative,
                    "kind": "file",
                    "symlink_target": None,
                    "bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    if not files:
        raise ValueError("runtime asset tree contains no files")
    names = {row["relative_path"] for row in files}
    if "config.json" not in names:
        raise ValueError("runtime asset tree lacks config.json")
    if not any("tokenizer" in Path(name).name for name in names):
        raise ValueError("runtime asset tree lacks tokenizer files")
    if not any(
        name.endswith((".safetensors", ".bin"))
        and "tokenizer" not in Path(name).name
        for name in names
    ):
        raise ValueError("runtime asset tree lacks model weight files")
    portable = {
        "root_kind": "symlink_dir" if root_is_symlink else "directory",
        "root_symlink_target": root_link_target,
        "files": files,
    }
    return {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        **portable,
        "file_count": len(files),
        "bytes": sum(int(row["bytes"]) for row in files),
        "tree_sha256": nested_sha256(portable),
    }


def portable_runtime_asset_tree(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "root_kind": record["root_kind"],
        "root_symlink_target": record.get("root_symlink_target"),
        "files": record["files"],
        "file_count": int(record["file_count"]),
        "bytes": int(record["bytes"]),
        "tree_sha256": record["tree_sha256"],
    }


def prepare_input_lock(
    *,
    repo_root: Path,
    e0_data_root: Path,
    output_sidecar: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    """Materialize the local sidecar; caller must run only after code lock."""

    if output_sidecar.exists() or output_manifest.exists():
        raise FileExistsError("input-lock output already exists")
    import torch
    from transformers import AutoTokenizer
    from rosetta.train.dataset_adapters import AlignedChatDataset
    from rosetta.utils.evaluate import set_default_chat_template
    from rosetta.utils.model_loading import resolve_model_path
    from script.evaluation.unified_evaluator import UnifiedEvaluator

    if torch.cuda.is_initialized():
        raise RuntimeError("prepare-input-lock refuses an initialized CUDA runtime")
    split = load_e0_design_lock(
        repo_root / "recipe/eval_recipe/fpct_e1/e1_data_split_manifest.json"
    )
    dev = verify_e0_dev_anchor(repo_root / E0_DEV_MANIFEST_RELATIVE, split)
    receiver_name = "Qwen/Qwen3-0.6B"
    sender_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    receiver_path = Path(resolve_model_path(receiver_name))
    sender_path = Path(resolve_model_path(sender_name))
    runtime_assets = {
        "receiver": {
            "model_id": receiver_name,
            **runtime_asset_tree(receiver_path),
        },
        "sender": {
            "model_id": sender_name,
            **runtime_asset_tree(sender_path),
        },
    }
    receiver = AutoTokenizer.from_pretrained(receiver_path)
    sender = AutoTokenizer.from_pretrained(sender_path)
    set_default_chat_template(receiver, receiver_name)
    set_default_chat_template(sender, sender_name)
    dimensions = _config_dimensions(receiver_path)

    reference_config = yaml.safe_load(
        (repo_root / "recipe/eval_recipe/fpct_e0/rendered/eval_2026072201_Y_FF_ai2-arc.yaml").read_text(
            encoding="utf-8"
        )
    )
    rosetta_config = reference_config["model"]["rosetta_config"]
    aligner = _build_aligner(receiver, sender, rosetta_config)
    dataset_cache: dict[tuple[str, str], Any] = {}
    items: list[dict[str, Any]] = []
    task_counts = {task: 0 for task in TASKS}
    for group, descriptor in sorted(dev["records"].items()):
        task = descriptor["task"]
        example = _load_task_example(
            task=task,
            descriptor=descriptor,
            data_root=e0_data_root,
            cache=dataset_cache,
        )
        question, choices = _question_choices(task, example)
        if canonical_content_sha256(question, choices) != group:
            raise ValueError("input-lock materialized content hash mismatch")
        expected_sample = canonical_sample_sha256(
            task, descriptor["subject"], descriptor["source_row_id"]
        )
        if expected_sample != descriptor["sample_sha256"]:
            raise ValueError("input-lock sample hash mismatch")
        formatter = UnifiedEvaluator.__new__(UnifiedEvaluator)
        formatter.dataset_name = task
        formatter.model_config = reference_config["model"]
        formatter.eval_config = {
            **reference_config["eval"],
            "dataset": task,
            "use_cot": False,
            "use_template": True,
        }
        prompt = formatter.format_example(example, use_cot=False)
        prompt_details = _prompt_alignment_details(aligner, prompt, top_k=4)
        rendered_sha = _sha256_bytes(prompt_details["slm_text"].encode("utf-8"))
        if rendered_sha != descriptor["rendered_prompt_sha256"]:
            raise ValueError("input-lock rendered prompt SHA mismatch")
        if _alignment_sha256(prompt_details) != descriptor["prompt_alignment_sha256"]:
            raise ValueError("input-lock prompt alignment SHA mismatch")
        answer = formatter.parse_answer(example)
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError("input-lock has no canonical A-D gold answer")
        gold_response = GOLD_RESPONSE_TEMPLATE.format(answer=answer)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": gold_response},
        ]
        raw_full_details = aligner.align_chat_messages_soft(
            messages,
            add_generation_prompt=False,
            return_details=True,
            apply_confidence_control=False,
            top_k=4,
        )
        full_details = aligner.sanitize_fpct_soft_alignment(
            raw_full_details,
            target_length=len(raw_full_details["slm_ids"]),
            source_length=len(raw_full_details["llm_ids"]),
        )
        aligned = AlignedChatDataset(
            [messages],
            aligner,
            max_length=32768,
            soft_alignment_top_k=4,
            fpct_alignment_sanitizer="certified_slot0_v1",
        )
        feature = aligned[0]
        _verify_feature_alignment(feature, full_details)
        instruction_end = _instruction_end(feature["labels"])
        provenance = feature_provenance(feature, full_details, gold_response)
        full_soft = full_details["soft_alignment"]
        alignment_lock = {
            "source_indices": full_soft["source_indices"],
            "source_weights": full_soft["source_weights"],
            "fpct_certified_mask": full_soft["fpct_certified_mask"],
            "fpct_offset_uncertified_mask": full_soft["fpct_offset_uncertified_mask"],
            "fpct_certification_reason": full_soft["fpct_certification_reason"],
        }
        topology = topology_contract(full_details, instruction_end)
        raw_topology = raw_topology_ledger(
            raw_details=raw_full_details,
            sanitized_details=full_details,
            instruction_end=instruction_end,
            task=task,
            sample_sha256=expected_sample,
            content_group_sha256=group,
            candidate_window=int(rosetta_config.get("soft_alignment_candidate_window", 0)),
        )
        formatter.model_config = reference_config["model"]
        prompt_prepared = formatter.prepare_model_inputs(
            prompt, receiver, torch.device("cpu"), "rosetta", sender, "generate"
        )
        prompt_inputs = prompt_prepared["inputs"]
        answer_queries = answer_query_contract(feature["labels"])
        certified_parents = certified_parent_contract(topology)
        expected_rows = expected_long_form_rows(
            answer_query_count=len(answer_queries),
            certified_parent_count=len(certified_parents),
            num_layers=dimensions["num_hidden_layers"],
            num_query_heads=dimensions["num_attention_heads"],
        )
        item = {
            "task": task,
            "sample_sha256": expected_sample,
            "content_group_sha256": group,
            "descriptor": descriptor,
            "gold_answer": answer,
            "gold_response": gold_response,
            "feature": feature,
            "alignment_lock": alignment_lock,
            "prompt_generation_inputs": prompt_inputs,
            "prompt_generation_inputs_sha256": nested_sha256(prompt_inputs),
            "provenance": provenance,
            "answer_queries": answer_queries,
            "certified_parents": certified_parents,
            "expected_long_form_rows": expected_rows,
            "raw_topology_ledger": raw_topology,
            "instruction_end": instruction_end,
            "rendered_prompt_sha256": rendered_sha,
        }
        item["item_semantic_sha256"] = nested_sha256(
            {key: value for key, value in item.items() if key != "feature"}
        )
        items.append(item)
        task_counts[task] += 1
    if task_counts != TASK_GROUP_COUNTS:
        raise ValueError("input-lock population differs from frozen 128/70/128")
    for role, model_path in (("receiver", receiver_path), ("sender", sender_path)):
        if portable_runtime_asset_tree(runtime_asset_tree(model_path)) != portable_runtime_asset_tree(runtime_assets[role]):
            raise ValueError(f"{role} runtime asset tree changed during CPU input locking")
    items.sort(key=lambda item: (item["task"], item["sample_sha256"]))
    by_task = {}
    for task in TASKS:
        members = [item for item in items if item["task"] == task]
        row_volume = expected_long_form_row_volume(members)
        raw_rows = [row for item in members for row in item["raw_topology_ledger"]]
        taxonomy_counts: dict[str, int] = {}
        for row in raw_rows:
            taxonomy_counts[row["taxonomy"]] = taxonomy_counts.get(row["taxonomy"], 0) + 1
        semantic_members = [
            {
                "sample_sha256": item["sample_sha256"],
                "content_group_sha256": item["content_group_sha256"],
                "item_semantic_sha256": item["item_semantic_sha256"],
            }
            for item in members
        ]
        task_group_contract = [
            {
                "content_group_sha256": group,
                "sample_sha256": [record["sample_sha256"]],
            }
            for group, record in dev["records"].items()
            if record["task"] == task
        ]
        by_task[task] = {
            "group_count": len(members),
            "members": semantic_members,
            "membership_sha256": task_membership_sha256(
                task_group_contract, semantic_members
            ),
            "row_template": row_template_attestation(
                members,
                num_layers=dimensions["num_hidden_layers"],
                num_query_heads=dimensions["num_attention_heads"],
                num_kv_heads=dimensions["num_key_value_heads"],
            ),
            "expected_long_form_rows": row_volume,
            "raw_topology": {
                "row_count": len(raw_rows),
                "ledger_sha256": nested_sha256(raw_rows),
                "raw_m_ge2_parent_count": len(raw_rows),
                "runtime_m_ge2_parent_count": sum(
                    row["runtime_candidate_count"] >= 2 for row in raw_rows
                ),
                "raw_m_distribution": {
                    str(value): sum(
                        row["raw_candidate_count"] == value for row in raw_rows
                    )
                    for value in range(2, 5)
                },
                "runtime_m_distribution": {
                    str(value): sum(
                        row["runtime_candidate_count"] == value
                        for row in raw_rows
                    )
                    for value in range(1, 5)
                },
                "offset_uncertified_parent_count": sum(
                    not row["certified"] for row in raw_rows
                ),
                "taxonomy_counts": taxonomy_counts,
                "candidate_window": 0,
                "competing_overlap_requires_explicit_evidence": True,
                "contains_model_output": False,
            },
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "split_role": "e0_design",
        "items": items,
        "dimensions": dimensions,
        "e1_pilot_consumed": False,
        "model_or_checkpoint_loaded": False,
        "cuda_initialized": False,
    }
    output_sidecar.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_sidecar.with_name(f".{output_sidecar.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output_sidecar)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_CPU_INPUTS_NO_MODEL_OUTPUT",
        "split_role": "e0_design",
        "source": {
            "split_sha256": split["sha256"],
            "dev_manifest_sha256": dev["sha256"],
            "e0_data_root": str(e0_data_root),
        },
        "tokenizers": {
            "receiver": {"name": receiver_name, "path": str(receiver_path), "files": _tokenizer_files(receiver_path)},
            "sender": {"name": sender_name, "path": str(sender_path), "files": _tokenizer_files(sender_path)},
        },
        "runtime_assets": runtime_assets,
        "dimensions": dimensions,
        "gold_response_template": GOLD_RESPONSE_TEMPLATE,
        "gold_response_template_sha256": _sha256_bytes(GOLD_RESPONSE_TEMPLATE.encode("utf-8")),
        "task_contract": by_task,
        "expected_long_form_rows_by_task": {
            task: {
                "count": by_task[task]["expected_long_form_rows"]["count"],
                "sum": by_task[task]["expected_long_form_rows"]["sum"],
            }
            for task in TASKS
        },
        "item_count": len(items),
        "sidecar": {
            "path": str(output_sidecar),
            "bytes": output_sidecar.stat().st_size,
            "sha256": sha256_file(output_sidecar),
        },
        "firewall": {
            "e1_pilot_consumed": False,
            "model_selection_consumed": False,
            "test_consumed": False,
            "model_or_checkpoint_loaded": False,
            "gpu_or_cuda_used": False,
        },
    }
    atomic_json(output_manifest, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--e0-data-root", type=Path, required=True)
    parser.add_argument("--output-sidecar", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = prepare_input_lock(
        repo_root=args.repo_root,
        e0_data_root=args.e0_data_root,
        output_sidecar=args.output_sidecar,
        output_manifest=args.output_manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
