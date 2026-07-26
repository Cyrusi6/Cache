#!/usr/bin/env python3
"""Concrete, fail-closed runtime backend for the FPCT-E1 E0-design audit.

This module is imported only after :mod:`fpct_e1_capture_runner` has verified
the prospective instrumentation gate and the immutable execution plan.  It
reuses the exact E0 evaluation formatter, ``AlignedChatDataset`` and
``RosettaDataCollator`` to run full-response teacher forcing.  The only accepted
model interchange is the bounded long-form capture contract; raw K/V tensors
are never read or written.

Every shard also reruns the frozen deterministic E0 generation/parser path and
emits only the current operator's group-level ``end_task_correct``.  C_post gold
log-probabilities and C_post correctness are owned and joined exclusively by the
executor; this backend cannot self-report a baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

import yaml

from script.experiment.fpct_e1_capture_runner import (
    ALLOWED_SPLIT_ROLE,
    CAPTURE_BACKEND_CONTRACT_VERSION,
    CAPTURE_STAGING_PROTOCOL_ID,
    CELL_SPEC,
    _audit_columns,
    _backend_projection_bytes,
    canonical_json_bytes,
    sha256_bytes,
)
from script.analysis.fpct_e1_streaming_verify import (
    PHYSICAL_CHUNK_ROWS,
    canonical_endpoint_id,
    encode_row_ordinal,
    endpoint_row_id,
    logical_row_id,
)


LONG_FORM_CONTRACT_VERSION = 2
GOLD_RESPONSE_TEMPLATE = "The correct answer is {answer}."
MAX_CANDIDATES = 4
PRIOR_ATOL = 2e-5

PRIMITIVE_FIELDS = (
    "batch_index",
    "layer",
    "query_head",
    "kv_head",
    "query_position",
    "parent_position",
    "candidate_count",
    "prior",
    "runtime_source_indices",
    "candidate_valid_mask",
    "candidate_slot_weights",
    "gamma",
    "source_d_k",
    "source_d_v",
    "source_energy_k",
    "source_energy_v",
    "fused_d_k",
    "fused_d_v",
    "fused_energy_k",
    "fused_energy_v",
    "candidate_logit_range",
    "candidate_logit_variance",
    "jensen_gap",
    "parent_attention_mass",
    "output_delta_l2",
)


class PrimitiveChunkSpool:
    """Ephemeral bounded tensor spool used between model capture and row join."""

    def __init__(
        self,
        *,
        expected_rows: int,
        num_query_heads: int,
        num_key_value_heads: int,
        root: Path | None = None,
    ) -> None:
        if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows <= 0:
            raise ValueError("primitive spool expected_rows must be positive")
        self.expected_rows = expected_rows
        if (
            isinstance(num_query_heads, bool)
            or not isinstance(num_query_heads, int)
            or isinstance(num_key_value_heads, bool)
            or not isinstance(num_key_value_heads, int)
            or num_query_heads <= 0
            or num_key_value_heads <= 0
            or num_query_heads % num_key_value_heads
        ):
            raise ValueError("primitive spool Hq/Hkv geometry is invalid")
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.root = Path(
            tempfile.mkdtemp(prefix="fpct-e1-primitives-", dir=root)
            if root is not None
            else tempfile.mkdtemp(prefix="fpct-e1-primitives-")
        )
        self.records: list[dict[str, Any]] = []
        self.row_count = 0
        self.max_chunk_rows = 0
        self.complete = False
        self.aborted = False
        self._previous_index: tuple[int, int, int, int, int] | None = None

    def write_primitive_chunk(self, chunk: Mapping[str, Any]) -> None:
        if self.complete or self.aborted:
            raise RuntimeError("primitive spool is already closed")
        import torch

        index = chunk.get("index")
        layer = chunk.get("layer")
        if (
            not isinstance(index, torch.Tensor)
            or index.ndim != 2
            or index.shape[1] != 4
            or isinstance(layer, bool)
            or not isinstance(layer, int)
        ):
            raise ValueError("primitive spool chunk index/layer contract mismatch")
        count = int(index.shape[0])
        if count <= 0 or count > PHYSICAL_CHUNK_ROWS:
            raise ValueError("primitive spool chunk exceeds 4096 rows")
        if self.row_count + count > self.expected_rows:
            raise RuntimeError("primitive spool emitted more than expected logical rows")
        if int(chunk.get("num_key_value_heads", -1)) != self.num_key_value_heads:
            raise ValueError("primitive spool Hkv changed")
        source_indices = chunk.get("source_indices")
        valid = chunk.get("valid")
        prior = chunk.get("prior")
        if (
            chunk.get("source_indices_certified") is not True
            or not isinstance(source_indices, torch.Tensor)
            or not isinstance(valid, torch.Tensor)
            or not isinstance(prior, torch.Tensor)
            or source_indices.shape != valid.shape
            or source_indices.shape != prior.shape
            or source_indices.ndim != 2
            or source_indices.shape[0] != count
            or source_indices.shape[1] != MAX_CANDIDATES
        ):
            raise ValueError("primitive spool lacks certified four-slot source identity")
        index_cpu = index.detach().cpu()
        first = (layer, *(int(value) for value in index_cpu[0]))
        last = (layer, *(int(value) for value in index_cpu[-1]))
        if first > last or (
            self._previous_index is not None and first <= self._previous_index
        ):
            raise ValueError("primitive spool chunks are not in frozen numeric order")
        previous = None
        for values in index_cpu:
            current = (layer, *(int(value) for value in values))
            if previous is not None and current <= previous:
                raise ValueError("primitive spool rows are duplicate or reordered")
            previous = current
        self._previous_index = last
        chunk_index = len(self.records)
        final = self.root / f"chunk_{chunk_index:08d}.pt"
        temporary = self.root / f".{final.name}.tmp"
        if final.exists() or temporary.exists():
            raise FileExistsError("primitive spool chunk path already exists")
        torch.save(dict(chunk), temporary)
        os.replace(temporary, final)
        self.records.append(
            {
                "chunk_index": chunk_index,
                "path": final,
                "row_count": count,
                "first": first,
                "last": last,
            }
        )
        self.row_count += count
        self.max_chunk_rows = max(self.max_chunk_rows, count)

    def finalize(self) -> dict[str, Any]:
        if self.aborted:
            raise RuntimeError("cannot finalize an aborted primitive spool")
        if self.row_count != self.expected_rows:
            raise RuntimeError(
                f"primitive spool row count mismatch: {self.row_count} != {self.expected_rows}"
            )
        self.complete = True
        return {
            "complete": True,
            "row_count": self.row_count,
            "chunk_count": len(self.records),
            "max_chunk_rows": self.max_chunk_rows,
            "physical_chunk_rows": PHYSICAL_CHUNK_ROWS,
            "stores_raw_kv": False,
        }

    def abort(self) -> None:
        self.aborted = True
        shutil.rmtree(self.root, ignore_errors=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def iter_primitives(self) -> Iterator[dict[str, Any]]:
        if not self.complete or self.aborted:
            raise RuntimeError("primitive spool is not complete")
        import torch

        emitted = 0
        for record in self.records:
            chunk = torch.load(record["path"], map_location="cpu", weights_only=False)
            index = chunk["index"]
            legal_values = chunk["valid"]
            for offset in range(int(index.shape[0])):
                batch, query_head, query_position, parent = (
                    int(value) for value in index[offset]
                )
                legal = legal_values[offset].bool()
                candidate_count = int(legal.sum())
                if candidate_count < 2:
                    raise ValueError("streamed primitive contains a non-ambiguous parent")
                row: dict[str, Any] = {
                    "batch_index": batch,
                    "layer": int(chunk["layer"]),
                    "query_head": query_head,
                    "kv_head": query_head
                    // (self.num_query_heads // self.num_key_value_heads),
                    "query_position": query_position,
                    "parent_position": parent,
                    "candidate_count": candidate_count,
                    "prior": [float(value) for value in chunk["prior"][offset][legal]],
                    "runtime_source_indices": [
                        int(value) for value in chunk["source_indices"][offset]
                    ],
                    "candidate_valid_mask": [bool(value) for value in legal],
                    "candidate_slot_weights": [
                        float(value) if bool(valid_value) else 0.0
                        for value, valid_value in zip(
                            chunk["prior"][offset], legal
                        )
                    ],
                    "gamma": [float(value) for value in chunk["gamma"][offset][legal]],
                    "output_delta_l2": float(chunk["output_delta_l2"][offset]),
                }
                row.update(
                    {
                        name: float(chunk["parent_geometry"][name][offset])
                        for name in (
                            "source_d_k", "source_d_v", "source_energy_k", "source_energy_v",
                            "fused_d_k", "fused_d_v", "fused_energy_k", "fused_energy_v",
                        )
                    }
                )
                row.update(
                    {
                        name: float(chunk["parent_metrics"][name][offset])
                        for name in (
                            "candidate_logit_range", "candidate_logit_variance",
                            "jensen_gap", "parent_attention_mass",
                        )
                    }
                )
                emitted += 1
                yield row
        if emitted != self.expected_rows:
            raise RuntimeError("primitive spool replay count differs from expected rows")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def canonical_content_sha256(question: str, choices: Sequence[str]) -> str:
    """Mirror the frozen FPCT-1B content-group projection exactly."""

    padded = [
        _normalize_text(choices[index]) if index < min(4, len(choices)) else ""
        for index in range(10)
    ]
    payload = {"question": _normalize_text(question), "choices": padded}
    return _sha256_bytes(_canonical_json(payload))


def canonical_sample_sha256(task: str, subject: str, question_id: str) -> str:
    return _sha256_bytes(
        _canonical_json(
            {"task": task, "subject": subject, "question_id": str(question_id)}
        )
    )


def _question_choices(task: str, example: Mapping[str, Any]) -> tuple[str, list[str]]:
    if task == "ai2-arc":
        question = str(example.get("question", ""))
        raw = example.get("choices", {})
    elif task == "openbookqa":
        question = str(example.get("question_stem", ""))
        raw = example.get("choices", {})
    elif task == "mmlu-redux":
        question = str(example.get("question", ""))
        return question, [str(value) for value in example.get("choices", [])[:4]]
    else:
        raise ValueError(f"unsupported E1 task: {task}")
    if isinstance(raw, Mapping):
        choices = [str(value) for value in raw.get("text", [])[:4]]
    elif isinstance(raw, list):
        choices = [
            str(value.get("text", "")) if isinstance(value, Mapping) else str(value)
            for value in raw[:4]
        ]
    else:
        choices = []
    return question, choices


def _to_python(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return {str(key): _to_python(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(child) for child in value]
    return value


def capture_primitives(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Legacy v1 helper retained only for bounded synthetic fixtures."""

    if report.get("stores_raw_kv") is not False:
        raise RuntimeError("capture report does not attest stores_raw_kv=false")
    if report.get("long_form_contract_version") != 1:
        raise RuntimeError("legacy long-form capture contract version mismatch")
    if report.get("long_form_incomplete_chunk_count") != 0:
        raise RuntimeError("bounded long-form capture contains incomplete chunks")
    raw = report.get("long_form_primitives")
    if not isinstance(raw, list):
        raise RuntimeError("capture report lacks bounded long_form_primitives")
    row_count = report.get("long_form_row_count")
    ceiling = report.get("max_long_form_rows")
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        raise RuntimeError("capture report has no integer long_form_row_count")
    if isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling <= 0:
        raise RuntimeError("capture report has no positive integer row ceiling")
    if row_count != len(raw):
        raise RuntimeError("capture long_form_row_count differs from materialized rows")
    if row_count > ceiling:
        raise RuntimeError("capture long-form row count exceeds the frozen ceiling")
    result: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise TypeError(f"long-form primitive {index} is not a mapping")
        missing = [name for name in PRIMITIVE_FIELDS if name not in value]
        if missing:
            raise RuntimeError(f"long-form primitive {index} lacks fields: {missing}")
        result.append({name: _to_python(value[name]) for name in PRIMITIVE_FIELDS})
    if not result:
        raise RuntimeError("capture report contains no certified m>=2 primitives")
    return result


def teacher_forced_capture_bounded(
    model: Any,
    forward_inputs: Mapping[str, Any],
    labels: Any,
    *,
    metadata: Mapping[str, Any],
    expected_long_form_rows: int,
    primitive_sink: PrimitiveChunkSpool,
) -> Any:
    """Run exact-count teacher forcing with bounded sink emission."""

    import torch

    if labels.ndim != 2:
        raise ValueError("labels must be [B,T]")
    if (
        isinstance(expected_long_form_rows, bool)
        or not isinstance(expected_long_form_rows, int)
        or expected_long_form_rows <= 0
    ):
        raise ValueError("expected_long_form_rows must be a positive integer")
    expected = torch.zeros_like(labels, dtype=torch.bool)
    expected[:, :-1] = labels[:, 1:] != -100
    for name in ("begin_fpct_capture", "end_fpct_capture", "fpct_teacher_forced_query_mask"):
        if not callable(getattr(model, name, None)):
            raise RuntimeError(f"model lacks explicit FPCT capture API: {name}")
    model_mask = model.fpct_teacher_forced_query_mask(labels)
    if not torch.equal(model_mask.detach().cpu(), expected.detach().cpu()):
        raise RuntimeError("teacher-forced query mask violates causal shift")
    model.begin_fpct_capture(
        mode="teacher_forced_response",
        metadata=dict(metadata),
        query_mask=model_mask,
        expected_long_form_rows=expected_long_form_rows,
        primitive_sink=primitive_sink,
        primitive_chunk_rows=PHYSICAL_CHUNK_ROWS,
        detail_mode="aggregate_only",
    )
    outputs = model(**dict(forward_inputs), labels=labels)
    report = model.end_fpct_capture()
    if report.get("long_form_contract_version") != LONG_FORM_CONTRACT_VERSION:
        raise RuntimeError("capture report streaming contract version mismatch")
    if report.get("expected_long_form_rows") != expected_long_form_rows:
        raise RuntimeError("capture report exact count differs from frozen input lock")
    if report.get("long_form_row_count") != expected_long_form_rows:
        raise RuntimeError("capture row count differs from the exact frozen row universe")
    if "long_form_primitives" in report:
        raise RuntimeError("streaming capture report materialized a primitive row list")
    receipt = report.get("long_form_stream")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("complete") is not True
        or receipt.get("row_count") != expected_long_form_rows
        or int(receipt.get("max_chunk_rows", 0)) > PHYSICAL_CHUNK_ROWS
    ):
        raise RuntimeError("capture primitive sink receipt is invalid")
    logits = outputs.logits
    shifted, eligible = labels[:, 1:], labels[:, 1:] != -100
    batch, query = torch.where(eligible)
    target, token = query + 1, shifted[batch, query]
    gold_logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)[batch, query, token]
    return SimpleNamespace(
        capture_report=report,
        logits=logits,
        loss=getattr(outputs, "loss", None),
        gold={
            "batch_index": batch,
            "query_position": query,
            "target_position": target,
            "target_token_id": token,
            "gold_logp": gold_logp,
        },
    )


def _compact_prior(weights: Sequence[Any], indices: Sequence[Any]) -> list[float]:
    prior = [
        float(weight)
        for weight, index in zip(weights, indices)
        if int(index) >= 0 and float(weight) > 0
    ]
    total = sum(prior)
    if not prior or not math.isfinite(total) or total <= 0:
        return []
    return [value / total for value in prior]


def topology_contract(details: Mapping[str, Any], instruction_end: int) -> dict[int, dict[str, Any]]:
    """Map sanitized parent positions to frozen candidate priors/topology."""

    soft = details.get("soft_alignment")
    if not isinstance(soft, Mapping):
        raise ValueError("sanitized alignment lacks soft_alignment")
    indices = soft.get("source_indices")
    weights = soft.get("source_weights")
    certified = soft.get("fpct_certified_mask")
    if not isinstance(indices, list) or not isinstance(weights, list) or not isinstance(certified, list):
        raise ValueError("sanitized alignment lacks certification metadata")
    if not (len(indices) == len(weights) == len(certified)):
        raise ValueError("sanitized alignment metadata lengths differ")
    result: dict[int, dict[str, Any]] = {}
    for parent, (row_indices, row_weights) in enumerate(zip(indices, weights)):
        if not isinstance(row_indices, list) or not isinstance(row_weights, list):
            raise ValueError("sanitized alignment candidate rows must be lists")
        if len(row_indices) != MAX_CANDIDATES or len(row_weights) != MAX_CANDIDATES:
            raise ValueError("sanitized alignment must preserve the frozen top-k=4 slots")
        normalized_indices = [int(value) for value in row_indices]
        normalized_weights = [float(value) for value in row_weights]
        if any(not math.isfinite(value) or value < 0 for value in normalized_weights):
            raise ValueError("sanitized alignment contains invalid slot mass")
        valid_mask = [
            index >= 0 and weight > 0
            for index, weight in zip(normalized_indices, normalized_weights)
        ]
        if any(
            (not valid) and (index != -1 or weight != 0.0)
            for index, weight, valid in zip(
                normalized_indices, normalized_weights, valid_mask
            )
        ):
            raise ValueError("sanitized alignment invalid slots are not canonical (-1,0)")
        legal_indices = [
            index for index, valid in zip(normalized_indices, valid_mask) if valid
        ]
        if len(set(legal_indices)) != len(legal_indices):
            raise ValueError("sanitized alignment contains duplicate legal source indices")
        prior = _compact_prior(row_weights, row_indices)
        count = len(prior)
        if count >= 2 and not bool(certified[parent]):
            raise ValueError("sanitizer retained an uncertified m>=2 parent")
        if count >= 2 and parent >= instruction_end:
            # include_response=false means response candidates are never eligible
            # transported memory for this experiment.
            continue
        result[parent] = {
            "candidate_count": count,
            "prior": prior,
            "candidate_indices": normalized_indices,
            "candidate_valid_mask": valid_mask,
            "candidate_slot_weights": normalized_weights,
            "statistical_weight": 1.0,
            "topology": "partition_compositional" if count >= 2 else "not_applicable",
            "within_instruction": parent < instruction_end,
        }
    return result


def _strict_distribution(values: Any, count: int, name: str) -> list[float]:
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{name} does not match candidate_count")
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise ValueError(f"{name} contains invalid mass")
    if abs(sum(result) - 1.0) > PRIOR_ATOL:
        raise ValueError(f"{name} does not sum to one")
    return result


def feature_provenance(
    feature: Mapping[str, Any],
    full_details: Mapping[str, Any],
    gold_response: str,
) -> dict[str, str]:
    """Hash the exact full-response input, alignment and labels."""

    input_payload = {
        "input_ids": _to_python(feature["input_ids"]),
        "model_padding_mask": _to_python(feature["model_padding_mask"]),
        "kv_cache_index": _to_python(feature["kv_cache_index"]),
    }
    soft = full_details["soft_alignment"]
    alignment_payload = {
        "source_indices": soft["source_indices"],
        "source_weights": soft["source_weights"],
        "fpct_certified_mask": soft["fpct_certified_mask"],
        "fpct_offset_uncertified_mask": soft["fpct_offset_uncertified_mask"],
        "fpct_certification_reason": soft["fpct_certification_reason"],
    }
    return {
        "input_sha256": _sha256_bytes(_canonical_json(input_payload) + b"\n"),
        "alignment_sha256": _sha256_bytes(
            _canonical_json(alignment_payload) + b"\n"
        ),
        "labels_sha256": _sha256_bytes(
            _canonical_json([int(value) for value in feature["labels"]]) + b"\n"
        ),
        "gold_response_sha256": _sha256_bytes(gold_response.encode("utf-8")),
    }


def iter_assemble_sample_rows(
    *,
    primitives: Iterator[Mapping[str, Any]],
    gold: Mapping[str, Any],
    topology: Mapping[int, Mapping[str, Any]],
    answer_queries: Sequence[Mapping[str, Any]],
    certified_parents: Sequence[Mapping[str, Any]],
    num_layers: int,
    num_query_heads: int,
    num_kv_heads: int,
    descriptor: Mapping[str, Any],
    shard: Mapping[str, Any],
    provenance: Mapping[str, str],
    end_task_correct: bool,
    require_complete: bool = True,
) -> Iterator[dict[str, Any]]:
    """Join current-operator primitives with sealed identity and gold tokens.

    No ``cpost_*`` field is accepted or returned.  Those fields are owned by
    the capture executor and mechanically joined from its immutable dependency.
    """

    batch_values = _to_python(gold["batch_index"])
    query_values = _to_python(gold["query_position"])
    target_values = _to_python(gold["target_position"])
    token_values = _to_python(gold["target_token_id"])
    logp_values = _to_python(gold["gold_logp"])
    gold_index: dict[tuple[int, int], tuple[int, int, float]] = {}
    for batch, query, target, token, logp in zip(
        batch_values, query_values, target_values, token_values, logp_values
    ):
        key = (int(batch), int(query))
        value = (int(target), int(token), float(logp))
        if key in gold_index:
            raise ValueError("duplicate teacher-forced gold query")
        if value[0] != key[1] + 1 or value[1] < 0 or not math.isfinite(value[2]):
            raise ValueError("invalid teacher-forced causal-shift gold row")
        gold_index[key] = value

    sample_values = descriptor.get("sample_sha256")
    if not isinstance(sample_values, list) or len(sample_values) != 1:
        raise ValueError("E0-design runtime requires exactly one canonical sample per group")
    sample_sha = str(sample_values[0])
    query_indices = {
        int(value["query_position"]): index
        for index, value in enumerate(answer_queries)
    }
    parent_indices = {
        int(value["parent_position"]): index
        for index, value in enumerate(certified_parents)
    }
    if len(query_indices) != len(answer_queries) or len(parent_indices) != len(
        certified_parents
    ):
        raise ValueError("frozen answer-query/parent sequence contains duplicates")
    endpoint = canonical_endpoint_id(
        int(shard["seed"]),
        shard["checkpoint_arm"],
        shard["inference_operator"],
        shard["cell"],
        shard["task"],
        float(shard["lambda_value"]),
    )
    expected_ordinal = 0
    for primitive in primitives:
        batch = int(primitive["batch_index"])
        query = int(primitive["query_position"])
        parent = int(primitive["parent_position"])
        if batch != 0:
            raise ValueError("one-sample runtime emitted a nonzero batch index")
        if (batch, query) not in gold_index:
            raise ValueError("mechanism primitive is not an eligible answer query")
        if parent not in topology or not topology[parent]["within_instruction"]:
            raise ValueError("mechanism primitive references non-prompt transported memory")
        if query not in query_indices or parent not in parent_indices:
            raise ValueError("mechanism primitive is outside frozen compact geometry")
        expected = topology[parent]
        count = int(primitive["candidate_count"])
        if count < 2 or count > MAX_CANDIDATES:
            raise ValueError("long-form capture must contain certified m>=2 parents only")
        if count != expected["candidate_count"]:
            raise ValueError("capture/alignment candidate count mismatch")
        prior = _strict_distribution(list(primitive["prior"]), count, "prior")
        expected_prior = list(expected["prior"])
        if any(abs(left - right) > PRIOR_ATOL for left, right in zip(prior, expected_prior)):
            raise ValueError("capture prior differs from frozen sanitized alignment")
        runtime_indices = [int(value) for value in primitive["runtime_source_indices"]]
        runtime_valid = [bool(value) for value in primitive["candidate_valid_mask"]]
        runtime_slot_weights = [float(value) for value in primitive["candidate_slot_weights"]]
        if (
            len(runtime_indices) != MAX_CANDIDATES
            or len(runtime_valid) != MAX_CANDIDATES
            or len(runtime_slot_weights) != MAX_CANDIDATES
            or runtime_indices != list(expected["candidate_indices"])
            or runtime_valid != list(expected["candidate_valid_mask"])
            or any(
                abs(left - right) > PRIOR_ATOL
                for left, right in zip(
                    runtime_slot_weights, expected["candidate_slot_weights"]
                )
            )
        ):
            raise ValueError("runtime candidate slot/mask identity differs from frozen alignment")
        gamma = _strict_distribution(list(primitive["gamma"]), count, "gamma")
        target, target_token_id, gold_logp = gold_index[(batch, query)]
        row = {
            "schema_version": 6,
            "split_role": ALLOWED_SPLIT_ROLE,
            "seed": int(shard["seed"]),
            "checkpoint_arm": shard["checkpoint_arm"],
            "inference_operator": shard["inference_operator"],
            "cell": shard["cell"],
            "task": shard["task"],
            "sample_sha256": sample_sha,
            "content_group_sha256": descriptor["content_group_sha256"],
            "input_sha256": provenance["input_sha256"],
            "alignment_sha256": provenance["alignment_sha256"],
            "labels_sha256": provenance["labels_sha256"],
            "gold_response_sha256": provenance["gold_response_sha256"],
            "layer": int(primitive["layer"]),
            "query_head": int(primitive["query_head"]),
            "kv_head": int(primitive["kv_head"]),
            "query_position": query,
            "target_position": target,
            "target_token_id": target_token_id,
            "parent_position": parent,
            "candidate_count": count,
            "topology": expected["topology"],
            "lambda_value": float(shard["lambda_value"]),
            "prior": prior,
            "candidate_indices": runtime_indices,
            "candidate_valid_mask": runtime_valid,
            "candidate_slot_weights": runtime_slot_weights,
            "statistical_weight": float(expected.get("statistical_weight", 1.0)),
            "gamma": gamma,
            "gold_logp": gold_logp,
            "end_task_correct": bool(end_task_correct),
        }
        for name in PRIMITIVE_FIELDS:
            if name not in {
                "batch_index",
                "layer",
                "query_head",
                "kv_head",
                "query_position",
                "parent_position",
                    "candidate_count",
                    "prior",
                    "runtime_source_indices",
                    "candidate_valid_mask",
                    "candidate_slot_weights",
                    "gamma",
            }:
                row[name] = float(primitive[name])
        ordinal = encode_row_ordinal(
            layer=row["layer"],
            query_head=row["query_head"],
            query_index=query_indices[query],
            parent_index=parent_indices[parent],
            num_layers=num_layers,
            num_query_heads=num_query_heads,
            answer_query_count=len(answer_queries),
            certified_parent_count=len(certified_parents),
        )
        if require_complete and ordinal != expected_ordinal:
            raise ValueError(
                f"streamed primitive order differs from ordinal contract: {ordinal} != {expected_ordinal}"
            )
        row["row_ordinal"] = ordinal
        row["logical_row_id"] = logical_row_id(row)
        row["endpoint_id"] = endpoint
        row["endpoint_row_id"] = endpoint_row_id(endpoint, row["logical_row_id"])
        expected_ordinal = expected_ordinal + 1 if require_complete else ordinal + 1
        yield row
    expected_count = (
        num_layers * num_query_heads * len(answer_queries) * len(certified_parents)
    )
    if require_complete and expected_ordinal != expected_count:
        raise RuntimeError("streamed runtime rows do not cover the full logical universe")


def assemble_sample_rows(
    *,
    primitives: Sequence[Mapping[str, Any]],
    gold: Mapping[str, Any],
    topology: Mapping[int, Mapping[str, Any]],
    descriptor: Mapping[str, Any],
    shard: Mapping[str, Any],
    provenance: Mapping[str, str],
    end_task_correct: bool,
) -> list[dict[str, Any]]:
    """Test-only materialized compatibility wrapper around the A4 iterator."""

    values = [dict(value) for value in primitives]
    if not values:
        return []
    query_values = sorted({int(value["query_position"]) for value in values})
    parent_values = sorted({int(value["parent_position"]) for value in values})
    answer_queries = [
        {
            "query_position": value,
            "target_position": value + 1,
            "target_token_id": int(
                _to_python(gold["target_token_id"])[
                    _to_python(gold["query_position"]).index(value)
                ]
            ),
        }
        for value in query_values
    ]
    certified_parents = [
        {"parent_position": value, **dict(topology[value])}
        for value in parent_values
    ]
    return list(
        iter_assemble_sample_rows(
            primitives=iter(values),
            gold=gold,
            topology=topology,
            answer_queries=answer_queries,
            certified_parents=certified_parents,
            num_layers=max(int(value["layer"]) for value in values) + 1,
            num_query_heads=max(int(value["query_head"]) for value in values) + 1,
            num_kv_heads=max(int(value["kv_head"]) for value in values) + 1,
            descriptor=descriptor,
            shard=shard,
            provenance=provenance,
            end_task_correct=end_task_correct,
            require_complete=False,
        )
    )


def _build_aligner(receiver_tokenizer: Any, sender_tokenizer: Any, config: Mapping[str, Any]) -> Any:
    from rosetta.model.aligner import AlignmentStrategy, TokenAligner

    return TokenAligner(
        slm_tokenizer=receiver_tokenizer,
        llm_tokenizer=sender_tokenizer,
        strategy=AlignmentStrategy(config["alignment_strategy"]),
        soft_alignment_score_mode=config["soft_alignment_score_mode"],
        soft_alignment_boundary_bonus=config["soft_alignment_boundary_bonus"],
        soft_alignment_boundary_tolerance=config["soft_alignment_boundary_tolerance"],
        soft_alignment_min_weight=config["soft_alignment_min_weight"],
        soft_alignment_confidence_mode=config["soft_alignment_confidence_mode"],
        soft_alignment_confidence_alpha=config["soft_alignment_confidence_alpha"],
        soft_alignment_confidence_floor=config["soft_alignment_confidence_floor"],
        soft_alignment_fallback_confidence=config["soft_alignment_fallback_confidence"],
        soft_alignment_confidence_control_mode=config.get(
            "soft_alignment_confidence_control_mode", "native"
        ),
        soft_alignment_confidence_constant_value=config.get(
            "soft_alignment_confidence_constant_value"
        ),
        soft_alignment_confidence_shuffle_seed=config.get(
            "soft_alignment_confidence_shuffle_seed", 0
        ),
        soft_alignment_reweight_mode=config.get("soft_alignment_reweight_mode", "none"),
        soft_alignment_reweight_strength=config.get("soft_alignment_reweight_strength", 1.0),
        soft_alignment_reweight_power=config.get("soft_alignment_reweight_power", 2.0),
        soft_alignment_candidate_window=config.get("soft_alignment_candidate_window", 0),
        learned_alignment_prior_mode=config.get("learned_alignment_prior_mode", "anchor"),
    )


def _verify_frozen_runtime_config(config: Mapping[str, Any], shard: Mapping[str, Any]) -> None:
    model = config.get("model", {})
    rosetta = model.get("rosetta_config", {})
    expected = {
        "base_model": "Qwen/Qwen3-0.6B",
        "teacher_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "attn_implementation": "eager",
        "alignment_strategy": "soft_span_overlap_v2",
        "soft_alignment_top_k": 4,
        "soft_alignment_score_mode": "uniform",
        "fpct_alignment_sanitizer": "certified_slot0_v1",
        "include_response": False,
        "fpct_operator": shard["inference_operator"],
    }
    mismatches = {
        name: (rosetta.get(name), value)
        for name, value in expected.items()
        if rosetta.get(name) != value
    }
    if mismatches:
        raise ValueError(f"frozen E0 runtime config mismatch: {mismatches}")
    if config.get("eval", {}).get("dataset") != shard["task"]:
        raise ValueError("frozen E0 runtime task mismatch")
    if config.get("eval", {}).get("use_cot") is not False:
        raise ValueError("E1 mechanism audit requires use_cot=false")
    if config.get("eval", {}).get("use_template") is not True:
        raise ValueError("E1 mechanism audit requires the canonical prompt template")


def _move_to_device(value: Any, device: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.to(device)
    except ImportError:
        pass
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _load_task_example(
    *,
    task: str,
    descriptor: Mapping[str, Any],
    data_root: Path,
    cache: MutableMapping[tuple[str, str], Any],
) -> Mapping[str, Any]:
    from rosetta.utils.dataset_loading import load_c2c_dataset

    if task == "ai2-arc":
        dataset_name, config_name = "allenai/ai2_arc", "ARC-Challenge"
    elif task == "openbookqa":
        dataset_name, config_name = "openbookqa", "main"
    elif task == "mmlu-redux":
        dataset_name, config_name = "edinburgh-dawg/mmlu-redux-2.0", str(
            descriptor["subject"]
        )
    else:
        raise ValueError(task)
    key = (task, config_name)
    if key not in cache:
        cache[key] = load_c2c_dataset(
            dataset_name,
            config_name=config_name,
            split="test",
            data_root_path=str(data_root),
        )
    dataset = cache[key]
    index = int(descriptor["evaluation_question_id"])
    if index < 0 or index >= len(dataset):
        raise ValueError("frozen E0 evaluation row index is outside materialized data")
    return dataset[index]


def _prompt_alignment_details(
    aligner: Any,
    prompt: str,
    *,
    top_k: int,
) -> Mapping[str, Any]:
    details = aligner.align_chat_messages_soft(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_details=True,
        enable_thinking=False,
        remove_last_surfix=False,
        top_k=top_k,
    )
    return aligner.sanitize_fpct_soft_alignment(
        details,
        target_length=len(details["slm_ids"]),
        source_length=len(details["llm_ids"]),
    )


def _alignment_sha256(details: Mapping[str, Any]) -> str:
    payload = {
        "slm_ids": details["slm_ids"],
        "llm_ids": details["llm_ids"],
        "source_indices": details["soft_alignment"]["source_indices"],
        "source_weights": details["soft_alignment"]["source_weights"],
    }
    return _sha256_bytes(_canonical_json(payload) + b"\n")


def _full_details(aligner: Any, messages: Sequence[Mapping[str, str]], top_k: int) -> Mapping[str, Any]:
    details = aligner.align_chat_messages_soft(
        list(messages),
        add_generation_prompt=False,
        return_details=True,
        apply_confidence_control=False,
        top_k=top_k,
    )
    return aligner.sanitize_fpct_soft_alignment(
        details,
        target_length=len(details["slm_ids"]),
        source_length=len(details["llm_ids"]),
    )


def _instruction_end(labels: Sequence[Any]) -> int:
    supervised = [index for index, value in enumerate(labels) if int(value) != -100]
    if not supervised:
        raise ValueError("teacher-forced example has no supervised response tokens")
    first = supervised[0]
    if any(int(value) != -100 for value in labels[:first]):
        raise ValueError("teacher-forced prompt label mask is not sealed")
    return first


def _verify_feature_alignment(feature: Mapping[str, Any], details: Mapping[str, Any]) -> None:
    observed = feature.get("soft_alignment", {})
    left_indices = _to_python(observed.get("source_indices"))
    right_indices = details["soft_alignment"]["source_indices"]
    if left_indices != right_indices:
        raise ValueError(
            "AlignedChatDataset differs from independently replayed source_indices"
        )
    left_weights = _to_python(observed.get("source_weights"))
    right_weights = details["soft_alignment"]["source_weights"]
    if len(left_weights) != len(right_weights) or any(
        len(left) != len(right)
        or any(abs(float(a) - float(b)) > PRIOR_ATOL for a, b in zip(left, right))
        for left, right in zip(left_weights, right_weights)
    ):
        raise ValueError(
            "AlignedChatDataset differs from independently replayed source_weights"
        )


def deterministic_end_task_correct(
    *,
    formatter: Any,
    model: Any,
    receiver_tokenizer: Any,
    sender_tokenizer: Any,
    prompt_inputs: Mapping[str, Any],
    answer: str,
    device: Any,
    generation_config: Mapping[str, Any],
) -> bool:
    """Rerun the exact frozen E0 deterministic generate/parser contract."""

    import torch

    if generation_config.get("do_sample") is not False or generation_config.get(
        "max_new_tokens"
    ) != 64:
        raise ValueError("E1 end-task outcome requires frozen greedy max_new_tokens=64")
    inputs = _move_to_device(dict(prompt_inputs), device)
    with torch.no_grad():
        output = model.generate(**dict(inputs), **dict(generation_config))
    if not isinstance(inputs.get("input_ids"), list):
        raise RuntimeError("E0 Rosetta generation did not use dual-tokenizer inputs")
    input_length = int(inputs["input_ids"][0].shape[1])
    generated = output[0][input_length:]
    content = receiver_tokenizer.decode(
        generated, skip_special_tokens=True
    ).strip("\n")
    predicted = formatter.extract_predicted_answer(content)
    return bool(predicted == answer)


def verify_expected_long_form_row_volume(
    items: Sequence[Mapping[str, Any]],
    task_contract: Mapping[str, Any],
    task_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute and exactly match the pre-data task row-volume contract."""

    from script.experiment.fpct_e1_prepare_input_lock import expected_long_form_row_volume

    observed = expected_long_form_row_volume(items)
    if task_contract.get("expected_long_form_rows") != observed:
        raise ValueError("input-lock task expected long-form row volume changed")
    expected_summary = {"count": observed["count"], "sum": observed["sum"]}
    if dict(task_summary) != expected_summary:
        raise ValueError("input-lock manifest task row-volume summary changed")
    return observed


def load_frozen_input_items(
    request: Mapping[str, Any], task: str
) -> list[dict[str, Any]]:
    """Load the CPU-prepared sidecar and verify every frozen byte/hash."""

    import torch

    manifest_path = Path(str(request.get("input_lock_manifest_path", "")))
    sidecar_path = Path(str(request.get("input_lock_sidecar_path", "")))
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("frozen E1 input-lock manifest/sidecar is unavailable")
    from script.experiment.fpct_e1_capture_runner import sha256_file

    if sha256_file(manifest_path) != request.get("input_lock_manifest_sha256"):
        raise ValueError("input-lock manifest SHA differs from execution plan")
    if sha256_file(sidecar_path) != request.get("input_lock_sidecar_sha256"):
        raise ValueError("input-lock sidecar SHA differs from execution plan")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("protocol_id")
        != "fpct_e1_e0_design_input_lock_v2_streaming"
        or manifest.get("status") != "GO_STREAMING_CPU_INPUT_LOCK_NO_MODEL_OUTPUT"
        or manifest.get("split_role") != ALLOWED_SPLIT_ROLE
    ):
        raise ValueError("input-lock manifest identity/firewall mismatch")
    streaming_contract = manifest.get("streaming_contract", {})
    if (
        streaming_contract.get("protocol_id")
        != "fpct_e1_mechanism_audit_v6_representation_preserving_streaming"
        or streaming_contract.get("physical_chunk_rows") != PHYSICAL_CHUNK_ROWS
        or streaming_contract.get("historical_cumulative_ceiling_operative") is not False
        or not isinstance(streaming_contract.get("geometry_lock"), Mapping)
        or not isinstance(streaming_contract.get("streaming_template_lock"), Mapping)
    ):
        raise ValueError("input-lock streaming contract/receipt is absent")
    expected = request.get("expected_row_template")
    task_contract = manifest.get("task_contract", {}).get(task, {})
    if task_contract.get("row_template") != expected:
        raise ValueError("input-lock task row-template attestation mismatch")
    if task_contract.get("membership_sha256") != request.get("membership_sha256"):
        raise ValueError("input-lock task membership SHA differs from execution plan")
    from script.experiment.fpct_e1_prepare_input_lock import (
        expected_long_form_rows,
        task_membership_sha256,
    )

    if task_membership_sha256(
        request["group_contract"], task_contract.get("members", [])
    ) != request.get("membership_sha256"):
        raise ValueError("input-lock manifest membership is not bound to the plan")
    payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    if (
        payload.get("protocol_id") != manifest["protocol_id"]
        or payload.get("split_role") != ALLOWED_SPLIT_ROLE
        or payload.get("model_or_checkpoint_loaded") is not False
        or payload.get("streaming_contract")
        != {
            "protocol_id": streaming_contract["protocol_id"],
            "schema_sha256": streaming_contract["schema_sha256"],
            "physical_chunk_rows": streaming_contract["physical_chunk_rows"],
            "expanded_logical_rows_present": False,
            "compact_geometry_only": True,
            "geometry_lock": streaming_contract["geometry_lock"],
            "streaming_template_lock": streaming_contract[
                "streaming_template_lock"
            ],
        }
    ):
        raise ValueError("input-lock sidecar identity/firewall mismatch")
    items = [dict(item) for item in payload.get("items", []) if item.get("task") == task]
    if len(items) != TASK_GROUP_COUNTS[task]:
        raise ValueError("input-lock task population differs from frozen count")
    group_contract = {
        row["content_group_sha256"]: row for row in request["group_contract"]
    }
    if {item["content_group_sha256"] for item in items} != set(group_contract):
        raise ValueError("input-lock and execution-plan group universes differ")
    for item in items:
        descriptor = group_contract[item["content_group_sha256"]]
        if descriptor["sample_sha256"] != [item["sample_sha256"]]:
            raise ValueError("input-lock sample identity differs from execution plan")
        locked_descriptor = item.get("descriptor", {})
        for name in (
            "task", "source_row_id", "subject", "evaluation_subject",
            "evaluation_question_id", "rendered_prompt_sha256",
            "prompt_alignment_sha256", "certified_candidate_count", "eligibility",
        ):
            if locked_descriptor.get(name) != descriptor.get(name):
                raise ValueError(f"input-lock descriptor differs from plan: {name}")
        if nested_input_sha256(item["prompt_generation_inputs"]) != item[
            "prompt_generation_inputs_sha256"
        ]:
            raise ValueError("input-lock prompt generation inputs changed")
        feature = item["feature"]
        input_payload = {
            "input_ids": _to_python(feature["input_ids"]),
            "model_padding_mask": _to_python(feature["model_padding_mask"]),
            "kv_cache_index": _to_python(feature["kv_cache_index"]),
        }
        observed = {
            "input_sha256": _sha256_bytes(_canonical_json(input_payload) + b"\n"),
            "alignment_sha256": _sha256_bytes(
                _canonical_json(item["alignment_lock"]) + b"\n"
            ),
            "labels_sha256": _sha256_bytes(
                _canonical_json([int(value) for value in feature["labels"]]) + b"\n"
            ),
            "gold_response_sha256": _sha256_bytes(
                item["gold_response"].encode("utf-8")
            ),
        }
        if observed != item["provenance"]:
            raise ValueError("input-lock item semantic provenance changed")
        observed_rows = expected_long_form_rows(
            answer_query_count=len(item.get("answer_queries", [])),
            certified_parent_count=len(item.get("certified_parents", [])),
            num_layers=int(manifest["dimensions"]["num_hidden_layers"]),
            num_query_heads=int(manifest["dimensions"]["num_attention_heads"]),
        )
        if item.get("expected_long_form_rows") != observed_rows:
            raise ValueError("input-lock item logical row count changed")
        if (
            item.get("Q_s") != len(item.get("answer_queries", []))
            or item.get("P_s") != len(item.get("certified_parents", []))
            or item.get("N_s") != observed_rows
            or item.get("expected_chunk_count")
            != math.ceil(observed_rows / PHYSICAL_CHUNK_ROWS)
        ):
            raise ValueError("input-lock compact geometry fields changed")
        semantic = {
            key: value
            for key, value in item.items()
            if key not in {"feature", "item_semantic_sha256"}
        }
        if nested_input_sha256(semantic) != item.get("item_semantic_sha256"):
            raise ValueError("input-lock item semantic SHA changed")
    semantic_members = [
        {
            "sample_sha256": item["sample_sha256"],
            "content_group_sha256": item["content_group_sha256"],
            "item_semantic_sha256": item["item_semantic_sha256"],
        }
        for item in items
    ]
    membership = task_membership_sha256(request["group_contract"], semantic_members)
    if membership != request.get("membership_sha256"):
        raise ValueError("input-lock sidecar membership SHA differs from manifest")
    if sorted(semantic_members, key=lambda row: row["sample_sha256"]) != sorted(
        task_contract.get("members", []), key=lambda row: row["sample_sha256"]
    ):
        raise ValueError("input-lock manifest member ledger differs from sidecar")
    verify_expected_long_form_row_volume(
        items,
        task_contract,
        manifest.get("expected_long_form_rows_by_task", {}).get(task, {}),
    )
    return _canonical_runtime_items(items)


def _canonical_runtime_items(
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Canonicalize any sidecar storage order to the frozen stream order."""

    ordered_items = sorted(
        (dict(item) for item in items), key=lambda item: item["sample_sha256"],
    )
    ordered_samples = [item["sample_sha256"] for item in ordered_items]
    if len(ordered_samples) != len(set(ordered_samples)):
        raise ValueError("input-lock runtime stream contains a duplicate sample")
    return ordered_items


def nested_input_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            _to_python(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def verify_runtime_assets_before_load(request: Mapping[str, Any]) -> dict[str, Any]:
    """Hash every pretrained runtime file before tokenizer/model/checkpoint load."""

    from rosetta.utils.model_loading import resolve_model_path
    from script.experiment.fpct_e1_capture_runner import sha256_file
    from script.experiment.fpct_e1_prepare_input_lock import (
        portable_runtime_asset_tree,
        runtime_asset_tree,
    )

    manifest_path = Path(str(request.get("input_lock_manifest_path", "")))
    if not manifest_path.is_file() or sha256_file(manifest_path) != request.get(
        "input_lock_manifest_sha256"
    ):
        raise ValueError("runtime asset verification lacks the frozen input manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = request.get("runtime_assets")
    if not isinstance(expected, Mapping) or expected != manifest.get("runtime_assets"):
        raise ValueError("plan/request runtime assets differ from the input-lock manifest")
    identities = {
        "receiver": "Qwen/Qwen3-0.6B",
        "sender": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    attestation: dict[str, Any] = {}
    for role, model_id in identities.items():
        frozen = expected.get(role)
        if not isinstance(frozen, Mapping) or frozen.get("model_id") != model_id:
            raise ValueError(f"frozen {role} runtime asset identity mismatch")
        actual_path = Path(resolve_model_path(model_id))
        actual = runtime_asset_tree(actual_path)
        if portable_runtime_asset_tree(actual) != portable_runtime_asset_tree(frozen):
            raise ValueError(f"{role} runtime asset tree changed before model load")
        attestation[role] = {
            "model_id": model_id,
            "runtime_path": str(actual_path),
            "tree_sha256": actual["tree_sha256"],
            "file_count": actual["file_count"],
            "bytes": actual["bytes"],
            "verified_before_model_or_tokenizer_load": True,
        }
    return attestation


def _validate_capture_resume_cursor(
    request: Mapping[str, Any], input_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind a runner-verified prefix to the compact sample geometry."""

    sample_order = [str(item.get("sample_sha256", "")) for item in input_items]
    if sample_order != sorted(sample_order) or len(sample_order) != len(set(sample_order)):
        raise ValueError("runtime capture items are not in unique sample-SHA order")
    cursor = request.get("resume_cursor")
    shard = request["shard"]
    if not isinstance(cursor, Mapping):
        raise ValueError("runtime capture request lacks a resume cursor")
    payload = {key: value for key, value in cursor.items() if key != "cursor_sha256"}
    if (
        cursor.get("schema_version") != 1
        or cursor.get("protocol_id") != CAPTURE_STAGING_PROTOCOL_ID
        or cursor.get("plan_sha256") != request["plan_sha256"]
        or cursor.get("shard_id") != shard["shard_id"]
        or cursor.get("cursor_sha256")
        != sha256_bytes(canonical_json_bytes(payload))
        or cursor.get("scientific_prefix_reverified") is not True
        or cursor.get("resume_only_from_first_incomplete_range") is not True
        or cursor.get("backend_projection_columns") != list(_audit_columns()[0])
        or cursor.get("partial_sample_backend_prefix_row_count")
        != cursor.get("next_row_ordinal")
        or not isinstance(
            cursor.get("partial_sample_backend_prefix_sha256"), str,
        )
        or len(cursor["partial_sample_backend_prefix_sha256"]) != 64
    ):
        raise ValueError("runtime capture resume cursor identity changed")
    completed_rows = cursor.get("completed_logical_rows")
    completed_samples = cursor.get("completed_sample_count")
    next_ordinal = cursor.get("next_row_ordinal")
    if (
        not isinstance(completed_rows, int)
        or not isinstance(completed_samples, int)
        or not isinstance(next_ordinal, int)
        or completed_rows < 0
        or not 0 <= completed_samples <= len(input_items)
        or next_ordinal < 0
    ):
        raise ValueError("runtime capture resume cursor range is invalid")
    prefix_rows = sum(
        int(item["expected_long_form_rows"])
        for item in input_items[:completed_samples]
    )
    if completed_samples == len(input_items):
        if (
            cursor.get("next_sample_sha256") is not None
            or next_ordinal != 0
            or completed_rows != prefix_rows
        ):
            raise ValueError("complete runtime capture cursor changed")
    else:
        next_item = input_items[completed_samples]
        if (
            cursor.get("next_sample_sha256") != next_item["sample_sha256"]
            or next_ordinal >= int(next_item["expected_long_form_rows"])
            or next_ordinal % PHYSICAL_CHUNK_ROWS != 0
            or completed_rows != prefix_rows + next_ordinal
        ):
            raise ValueError("runtime capture cursor is not the first incomplete chunk")
    return dict(cursor)


class _RecomputedPrefixVerifier:
    """Online exact backend-column digest used before yielding a resume suffix."""

    def __init__(
        self, cursor: Mapping[str, Any],
        channel: MutableMapping[str, Any],
    ) -> None:
        self.cursor = cursor
        self.channel = channel
        self.columns = _audit_columns()[0]
        self.digest = hashlib.sha256()
        self.count = 0
        self.finalized = False

    def observe(self, row: Mapping[str, Any]) -> None:
        if self.finalized:
            raise RuntimeError("resume prefix verifier observed a row after finalize")
        if self.count >= int(
            self.cursor["partial_sample_backend_prefix_row_count"]
        ):
            raise RuntimeError("resume prefix verifier observed too many rows")
        self.digest.update(_backend_projection_bytes(row, self.columns))
        self.count += 1

    def finalize(self) -> None:
        if self.finalized:
            return
        observed_sha = self.digest.hexdigest()
        exact = (
            self.count
            == int(self.cursor["partial_sample_backend_prefix_row_count"])
            and observed_sha
            == self.cursor["partial_sample_backend_prefix_sha256"]
        )
        self.channel.update({
            "status": (
                "GO_EXACT_PREFIX_MATCH" if exact
                else "FAILED_PREFIX_MISMATCH"
            ),
            "observed_row_count": self.count,
            "observed_backend_projection_sha256": observed_sha,
            "exact_match": exact,
        })
        self.finalized = True
        if not exact:
            raise RuntimeError(
                "recomputed runtime prefix differs from immutable prefix"
            )


def _iter_real_rows(
    *,
    request: Mapping[str, Any],
    model: Any,
    receiver_tokenizer: Any,
    sender_tokenizer: Any,
    formatter: Any,
    collator: Any,
    device: Any,
    generation_config: Mapping[str, Any],
    input_items: Sequence[Mapping[str, Any]],
    resume_cursor: Mapping[str, Any],
    resume_prefix_verification: MutableMapping[str, Any],
) -> Iterator[dict[str, Any]]:
    import torch

    shard = request["shard"]
    descriptors = {row["content_group_sha256"]: row for row in request["group_contract"]}
    completed_samples = int(resume_cursor["completed_sample_count"])
    for item_index, item in enumerate(input_items):
        if item_index < completed_samples:
            # Runner has recursively replayed and reverified these immutable
            # chunks before this backend/model was loaded.
            continue
        descriptor = descriptors[item["content_group_sha256"]]
        sample_values = descriptor["sample_sha256"]
        feature = item["feature"]
        topology = {
            int(parent["parent_position"]): {
                **parent,
                "within_instruction": True,
            }
            for parent in item["certified_parents"]
        }
        provenance = item["provenance"]
        answer = item["gold_answer"]
        end_task_correct = deterministic_end_task_correct(
            formatter=formatter,
            model=model,
            receiver_tokenizer=receiver_tokenizer,
            sender_tokenizer=sender_tokenizer,
            prompt_inputs=item["prompt_generation_inputs"],
            answer=answer,
            device=device,
            generation_config=generation_config,
        )
        batch = collator([feature])
        labels = batch.pop("labels")
        batch = _move_to_device(batch, device)
        labels = labels.to(device)
        batch["use_cache"] = True
        metadata = {
            "split_role": ALLOWED_SPLIT_ROLE,
            "task": shard["task"],
            "sample_sha256": sample_values[0],
            "content_group_sha256": descriptor["content_group_sha256"],
            "gold_response_template_sha256": _sha256_bytes(
                GOLD_RESPONSE_TEMPLATE.encode("utf-8")
            ),
        }
        expected_rows = int(item["expected_long_form_rows"])
        spool = PrimitiveChunkSpool(
            expected_rows=expected_rows,
            num_query_heads=16,
            num_key_value_heads=8,
        )
        try:
            with torch.no_grad():
                observation = teacher_forced_capture_bounded(
                    model,
                    batch,
                    labels,
                    metadata=metadata,
                    expected_long_form_rows=expected_rows,
                    primitive_sink=spool,
                )
            emitted = 0
            skip_before = (
                int(resume_cursor["next_row_ordinal"])
                if item_index == completed_samples else 0
            )
            prefix_verifier = _RecomputedPrefixVerifier(
                resume_cursor, resume_prefix_verification,
            )
            if skip_before == 0:
                prefix_verifier.finalize()
            for row in iter_assemble_sample_rows(
                primitives=spool.iter_primitives(),
                gold=observation.gold,
                topology=topology,
                answer_queries=item["answer_queries"],
                certified_parents=item["certified_parents"],
                num_layers=28,
                num_query_heads=16,
                num_kv_heads=8,
                descriptor=descriptor,
                shard=shard,
                provenance=provenance,
                end_task_correct=end_task_correct,
            ):
                if int(row["row_ordinal"]) != emitted:
                    raise RuntimeError("runtime sample row ordinal is not contiguous")
                emitted += 1
                if int(row["row_ordinal"]) < skip_before:
                    # A partially completed sample still executes once, but
                    # its already-committed prefix is checked and never
                    # re-emitted to the sink.
                    prefix_verifier.observe(row)
                    continue
                prefix_verifier.finalize()
                yield row
            if emitted != expected_rows:
                raise RuntimeError("certified E0-design sample emitted an incomplete row stream")
        finally:
            spool.cleanup()


def run_capture_backend(request: Mapping[str, Any]) -> dict[str, Any]:
    """Load one exact task/checkpoint/operator shard after all external gates."""

    if request.get("contract_version") != CAPTURE_BACKEND_CONTRACT_VERSION:
        raise ValueError("capture backend request contract mismatch")
    if request.get("split_role") != ALLOWED_SPLIT_ROLE:
        raise ValueError("data firewall: runtime backend accepts E0-design only")
    shard = request.get("shard")
    if not isinstance(shard, Mapping) or shard.get("cell") not in CELL_SPEC:
        raise ValueError("runtime backend shard identity is invalid")
    lambda_value = float(shard.get("lambda_value", -1.0))
    if shard.get("inference_operator") == "c_post":
        if lambda_value != 0.0:
            raise ValueError("C_post runtime is only valid at lambda=0")
    elif shard.get("inference_operator") == "f":
        if lambda_value not in {0.0, 0.25, 0.5, 1.0, 2.0}:
            raise ValueError("F centered-lambda value is outside the frozen grid")
    else:
        raise ValueError("runtime operator must be c_post or f")
    group_contract = request.get("group_contract")
    if not isinstance(group_contract, list) or not group_contract:
        raise ValueError("runtime backend has no frozen group contract")
    if any(row.get("task", shard["task"]) != shard["task"] for row in group_contract):
        raise ValueError("runtime backend group contract mixes tasks")
    e0_data_root = Path(str(request.get("e0_data_root", "")))
    if not e0_data_root.is_dir():
        raise FileNotFoundError("runtime E0 dev-data root is unavailable")
    config_path = Path(str(request.get("e0_config_path", "")))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _verify_frozen_runtime_config(config, shard)
    config["model"]["rosetta_config"]["checkpoints_dir"] = request[
        "checkpoint_path"
    ]
    config["model"]["rosetta_config"]["fpct_instrumentation"] = True
    config["model"]["rosetta_config"]["projector_load_mode"] = "strict_attested"
    config["eval"]["data_root"] = str(e0_data_root)
    config["eval"]["gpu_ids"] = [0]
    # Fail closed on the entire CPU-prepared input lock before loading either
    # pretrained model or any projector checkpoint.
    runtime_asset_attestation = verify_runtime_assets_before_load(request)
    input_items = load_frozen_input_items(request, shard["task"])
    resume_cursor = _validate_capture_resume_cursor(request, input_items)
    resume_prefix_verification: MutableMapping[str, Any] = {
        "schema_version": 1,
        "protocol_id": CAPTURE_STAGING_PROTOCOL_ID,
        "cursor_sha256": resume_cursor["cursor_sha256"],
        "expected_row_count": resume_cursor[
            "partial_sample_backend_prefix_row_count"
        ],
        "expected_backend_projection_sha256": resume_cursor[
            "partial_sample_backend_prefix_sha256"
        ],
        "status": "PENDING_RECOMPUTATION",
        "observed_row_count": None,
        "observed_backend_projection_sha256": None,
        "exact_match": False,
    }
    from script.experiment.fpct_e1_prepare_input_lock import expected_long_form_row_volume

    long_form_row_volume_attestation = {
        **expected_long_form_row_volume(input_items),
        "source": "frozen_cpu_input_lock",
        "passed_explicitly_per_item": True,
        "exact_row_count_required": True,
        "verified_before_model_load": True,
    }

    import torch
    from transformers import AutoTokenizer
    from rosetta.train.dataset_adapters import RosettaDataCollator
    from rosetta.utils.evaluate import (
        load_rosetta_model,
        set_default_chat_template,
    )
    from rosetta.utils.model_loading import resolve_model_path
    from script.evaluation.unified_evaluator import UnifiedEvaluator

    device_name = os.environ.get("FPCT_E1_DEVICE", "cuda:0")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("FPCT-E1 runtime requested CUDA but CUDA is unavailable")
    model, receiver_tokenizer = load_rosetta_model(
        config["model"],
        config["eval"],
        device,
        config["model"].get("generation_config", {}),
    )
    projector_load_attestation = getattr(model, "_projector_load_attestation", None)
    if (
        not isinstance(projector_load_attestation, Mapping)
        or projector_load_attestation.get("mode") != "strict_attested"
        or not projector_load_attestation.get("records")
        or any(
            record.get("missing_keys") or record.get("unexpected_keys")
            for record in projector_load_attestation.get("records", [])
        )
    ):
        raise RuntimeError("E1 projector strict-load attestation is absent or nonempty")
    if getattr(model, "fpct_operator", None) != shard["inference_operator"]:
        raise RuntimeError("loaded model operator differs from frozen shard")
    if shard["inference_operator"] == "f":
        setter = getattr(model, "set_fpct_centered_lambda", None)
        if not callable(setter):
            raise RuntimeError("loaded F model lacks frozen centered-lambda API")
        setter(lambda_value)
        if float(getattr(model, "fpct_centered_lambda", -1.0)) != lambda_value:
            raise RuntimeError("loaded model did not bind the frozen centered lambda")
    if not callable(getattr(model, "begin_fpct_capture", None)) or not callable(
        getattr(model, "end_fpct_capture", None)
    ):
        raise RuntimeError("loaded model lacks explicit E1 capture lifecycle")
    sender_name = config["model"]["rosetta_config"]["teacher_model"]
    sender_tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(sender_name))
    set_default_chat_template(sender_tokenizer, sender_name)
    formatter = UnifiedEvaluator.__new__(UnifiedEvaluator)
    formatter.dataset_name = shard["task"]
    formatter.model_config = config["model"]
    formatter.eval_config = config["eval"]
    formatter.generation_config = dict(config["model"].get("generation_config", {}))
    collator = RosettaDataCollator(
        receiver_tokenizer,
        sender_tokenizer,
        max_length=32768,
        aligner=None,
        do_alignment=False,
    )
    rows = _iter_real_rows(
        request=request,
        model=model,
        receiver_tokenizer=receiver_tokenizer,
        sender_tokenizer=sender_tokenizer,
        formatter=formatter,
        collator=collator,
        device=device,
        generation_config=formatter.generation_config,
        input_items=input_items,
        resume_cursor=resume_cursor,
        resume_prefix_verification=resume_prefix_verification,
    )
    return {
        "contract_version": CAPTURE_BACKEND_CONTRACT_VERSION,
        "rows": rows,
        "resume_prefix_verification": resume_prefix_verification,
        "attestation": {
            "split_role": ALLOWED_SPLIT_ROLE,
            "capture_mode": "teacher_forced_response",
            "causal_shift_verified": True,
            "stores_raw_kv": False,
            "long_form_contract_version": LONG_FORM_CONTRACT_VERSION,
            "plan_sha256": request["plan_sha256"],
            "shard_id": shard["shard_id"],
            "execution_sha": request["execution_sha"],
            "image_digest": request["image_digest"],
            "checkpoint_tree_sha256": shard["checkpoint_tree_sha256"],
            "membership_sha256": request["membership_sha256"],
            "resume_cursor_sha256": resume_cursor["cursor_sha256"],
            "resume_completed_logical_rows": resume_cursor[
                "completed_logical_rows"
            ],
            "resume_next_sample_sha256": resume_cursor[
                "next_sample_sha256"
            ],
            "resume_next_row_ordinal": resume_cursor["next_row_ordinal"],
            "gold_response_template": GOLD_RESPONSE_TEMPLATE,
            "gold_response_template_sha256": _sha256_bytes(
                GOLD_RESPONSE_TEMPLATE.encode("utf-8")
            ),
            "correctness_semantics": (
                "current operator deterministic E0-compatible generated-answer "
                "correctness repeated on rows; not teacher-forced token accuracy; "
                "C_post baseline is executor-owned"
            ),
            "backend_supplied_cpost_fields": False,
            "runtime_assets": runtime_asset_attestation,
            "projector_load": projector_load_attestation,
            "expected_long_form_rows": long_form_row_volume_attestation,
            "e1_pilot_consumed": False,
            "model_selection_consumed": False,
            "test_consumed": False,
        },
    }


__all__ = [
    "GOLD_RESPONSE_TEMPLATE",
    "LONG_FORM_CONTRACT_VERSION",
    "PrimitiveChunkSpool",
    "assemble_sample_rows",
    "canonical_content_sha256",
    "canonical_sample_sha256",
    "capture_primitives",
    "iter_assemble_sample_rows",
    "teacher_forced_capture_bounded",
    "verify_expected_long_form_row_volume",
    "deterministic_end_task_correct",
    "feature_provenance",
    "run_capture_backend",
    "topology_contract",
]
