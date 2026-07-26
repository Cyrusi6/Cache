#!/usr/bin/env python3
"""Offline FPCT-E1 mechanism audit and centered-lambda reference utilities.

This module deliberately does not load a tokenizer, dataset, model, or checkpoint.
The model-side capture lifecycle produces bounded long-form rows; this module
validates, derives, aggregates, and verifies those rows.  Its split firewall only
accepts the already-open E0-design role.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch

from script.analysis.fpct_e1_streaming_verify import (
    PROTOCOL_ID,
    SCHEMA_VERSION,
    canonical_endpoint_id,
    endpoint_row_id as expected_endpoint_row_id,
    logical_row_id as expected_logical_row_id,
)


SPLIT_ROLE = "e0_design"
TASKS = ("ai2-arc", "openbookqa", "mmlu-redux")
CHECKPOINT_ARMS = ("c_post_trained", "f_trained")
INFERENCE_OPERATORS = ("c_post", "f")
CELL_MAP = {
    ("c_post_trained", "c_post"): "Y_CC",
    ("c_post_trained", "f"): "Y_CF",
    ("f_trained", "c_post"): "Y_FC",
    ("f_trained", "f"): "Y_FF",
}
LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
TOPOLOGIES = (
    "not_applicable",
    "partition_compositional",
    "competing_overlap",
    "boundary_fallback",
    "neighbor_expansion",
    "taxonomy_unresolved",
)
EPSILON = 1e-12
FLOAT32_ATOL = 2e-5
PARQUET_BATCH_ROWS = 4096
MAX_SYNTHETIC_REFERENCE_ROWS = 2 * PARQUET_BATCH_ROWS
MAX_STAGE_PARTITIONS = 108
MAX_E0_DESIGN_SAMPLES = 326

PARTITION_COLUMNS = (
    "seed",
    "checkpoint_arm",
    "inference_operator",
    "cell",
    "task",
    "lambda_value",
)
SAMPLE_COLUMNS = ("sample_sha256", "content_group_sha256")
FUNCTIONAL_SORT_COLUMNS = (
    "endpoint_id",
    "sample_sha256",
    "row_ordinal",
    "endpoint_row_id",
)

OUTPUT_NAMES = {
    "rows": "e1_mechanism_rows.parquet",
    "summary": "e1_mechanism_summary.json",
    "layer_head": "e1_layer_head_summary.csv",
    "contraction": "e1_projector_contraction.csv",
    "topology": "e1_candidate_topology.csv",
    "lambda": "e1_centered_lambda_summary.csv",
}

RAW_TOPOLOGY_OUTPUT_NAMES = {
    "jsonl": "e1_raw_topology_ledger.jsonl",
    "parquet": "e1_raw_topology_ledger.parquet",
    "aggregate_csv": "e1_raw_topology_aggregates.csv",
    "aggregate_json": "e1_raw_topology_aggregates.json",
    "manifest": "e1_raw_topology_manifest.json",
}

RAW_CANDIDATE_COLUMNS = (
    "slot",
    "source_index",
    "source_token_id",
    "source_offset",
    "source_span",
    "intersection",
    "intersection_length",
    "origin",
    "raw_weight",
    "runtime_retained",
    "runtime_weight",
    "complete_receiver_explanation",
)

RAW_TOPOLOGY_COLUMNS = (
    "schema_version",
    "split_role",
    "task",
    "sample_sha256",
    "content_group_sha256",
    "parent_position",
    "receiver_token_id",
    "receiver_offset",
    "receiver_span",
    "raw_candidate_count",
    "runtime_candidate_count",
    "raw_candidate_indices",
    "runtime_candidate_indices",
    "raw_weights",
    "runtime_weights",
    "candidates",
    "certified",
    "offset_uncertified",
    "certification_reason",
    "taxonomy",
    "candidate_window",
    "duplicate_or_overlap_alias",
    "runtime_functional_eligible",
    "functional_metrics_present",
    "span_geometry_sha256",
)

RAW_AGGREGATE_COLUMNS = (
    "schema_version",
    "task",
    "taxonomy",
    "raw_m",
    "runtime_m",
    "parent_count",
    "sample_count",
    "content_group_count",
    "raw_candidate_atom_count",
    "runtime_candidate_atom_count",
    "raw_extra_slot_count",
    "runtime_extra_slot_count",
    "raw_minus_runtime_candidate_count",
    "raw_minus_runtime_extra_slot_count",
    "certified_parent_count",
    "offset_uncertified_parent_count",
    "functional_metric_contract",
)

INPUT_COLUMNS = (
    "schema_version",
    "split_role",
    "seed",
    "checkpoint_arm",
    "inference_operator",
    "cell",
    "task",
    "sample_sha256",
    "content_group_sha256",
    "input_sha256",
    "alignment_sha256",
    "labels_sha256",
    "gold_response_sha256",
    "layer",
    "query_head",
    "kv_head",
    "query_position",
    "target_position",
    "target_token_id",
    "parent_position",
    "candidate_count",
    "topology",
    "row_ordinal",
    "logical_row_id",
    "endpoint_id",
    "endpoint_row_id",
    "lambda_value",
    "prior",
    "candidate_indices",
    "candidate_valid_mask",
    "candidate_slot_weights",
    "statistical_weight",
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
    "gold_logp",
    "cpost_gold_logp",
    "end_task_correct",
    "cpost_end_task_correct",
)

DERIVED_COLUMNS = (
    "gamma_kl_prior",
    "gamma_tv_prior",
    "gamma_query_variance",
    "posterior_top1_candidate",
    "posterior_top1_changed",
    "normalized_source_d_k",
    "normalized_source_d_v",
    "normalized_fused_d_k",
    "normalized_fused_d_v",
    "projector_retention_k",
    "projector_retention_v",
    "delta_gold_logp",
    "end_task_accuracy_flip",
)

OUTPUT_COLUMNS = INPUT_COLUMNS + DERIVED_COLUMNS

MEAN_COLUMNS = (
    "candidate_count",
    "source_d_k",
    "source_d_v",
    "fused_d_k",
    "fused_d_v",
    "normalized_source_d_k",
    "normalized_source_d_v",
    "normalized_fused_d_k",
    "normalized_fused_d_v",
    "projector_retention_k",
    "projector_retention_v",
    "candidate_logit_range",
    "candidate_logit_variance",
    "gamma_kl_prior",
    "gamma_tv_prior",
    "gamma_query_variance",
    "jensen_gap",
    "parent_attention_mass",
    "output_delta_l2",
    "delta_gold_logp",
)

PARENT_GEOMETRY_METRICS = (
    "candidate_count",
    "topology",
    "source_d_k",
    "source_d_v",
    "source_energy_k",
    "source_energy_v",
    "fused_d_k",
    "fused_d_v",
    "fused_energy_k",
    "fused_energy_v",
    "normalized_source_d_k",
    "normalized_source_d_v",
    "normalized_fused_d_k",
    "normalized_fused_d_v",
    "projector_retention_k",
    "projector_retention_v",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _finite_float(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _integer(value: Any, name: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _sha256(value: Any, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return result


def _grid_value(value: Any) -> float:
    result = _finite_float(value, "lambda_value", nonnegative=True)
    matches = [item for item in LAMBDA_GRID if abs(item - result) <= 1e-12]
    if len(matches) != 1:
        raise ValueError(f"lambda_value must be in the frozen grid {LAMBDA_GRID}")
    return matches[0]


@dataclass
class RunningMoments:
    """Scalar Welford accumulator used by query-level online reductions."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        value = _finite_float(value, "running value")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count else 0.0


class GammaQueryAccumulator:
    """Online posterior variance across answer queries; never stores raw KV."""

    def __init__(self, candidate_count: int):
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        self._moments = [RunningMoments() for _ in range(candidate_count)]
        self._top1: set[int] = set()

    def update(self, gamma: Sequence[float]) -> None:
        if len(gamma) != len(self._moments):
            raise ValueError("posterior dimension changed across answer queries")
        for accumulator, value in zip(self._moments, gamma):
            accumulator.update(float(value))
        self._top1.add(max(range(len(gamma)), key=lambda index: (gamma[index], -index)))

    @property
    def count(self) -> int:
        return self._moments[0].count

    @property
    def mean_candidate_variance(self) -> float:
        return sum(item.variance for item in self._moments) / len(self._moments)

    @property
    def top1_changed(self) -> bool:
        return len(self._top1) > 1


def canonical_prior(prior: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask then L1-normalize a candidate prior (FP32, or FP64 for its oracle)."""

    if prior.shape != valid.shape:
        raise ValueError("prior and valid shapes differ")
    working = prior if prior.dtype == torch.float64 else prior.float()
    legal = valid.bool() & torch.isfinite(working) & (working > 0)
    illegal = (~torch.isfinite(working)) | (working < 0) | ((working > 0) & ~valid.bool())
    if bool(illegal.any()):
        raise ValueError("illegal candidate prior")
    mass = torch.where(legal, working, torch.zeros_like(working))
    total = mass.sum(dim=-1, keepdim=True)
    normalized = torch.where(total > 0, mass / total.clamp_min(torch.finfo(torch.float32).tiny), mass)
    return normalized, legal


def centered_candidates(
    candidates: torch.Tensor,
    prior: torch.Tensor,
    valid: torch.Tensor,
    lambda_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply X_lambda = Xbar + lambda * (X-Xbar) on the candidate axis."""

    lambda_value = _grid_value(lambda_value)
    if candidates.shape[-2] != prior.shape[-1]:
        raise ValueError("candidate/prior shapes are incompatible")
    normalized, legal = canonical_prior(prior, valid)
    if candidates.ndim == prior.ndim + 1 and candidates.shape[:-2] == prior.shape[:-1]:
        pass
    elif (
        candidates.ndim == prior.ndim + 2
        and candidates.shape[0] == prior.shape[0]
        and candidates.shape[-3] == prior.shape[-2]
    ):
        normalized = normalized.unsqueeze(1)
        legal = legal.unsqueeze(1)
    else:
        raise ValueError("candidate/prior shapes are incompatible")
    weight = normalized.to(candidates.dtype).unsqueeze(-1)
    mean = (candidates * weight).sum(dim=-2)
    centered = mean.unsqueeze(-2) + lambda_value * (candidates - mean.unsqueeze(-2))
    # Invalid values remain mathematically irrelevant and are zeroed to prevent accidental use.
    centered = torch.where(legal.unsqueeze(-1), centered, torch.zeros_like(centered))
    return centered, mean


def _masked_softmax(logits: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    masked = torch.where(active, logits, torch.full_like(logits, -torch.inf))
    any_active = active.any(dim=-1, keepdim=True)
    safe = torch.where(any_active, masked, torch.zeros_like(masked))
    working = safe if safe.dtype == torch.float64 else safe.float()
    probability = torch.softmax(working, dim=-1)
    probability = torch.where(active, probability, torch.zeros_like(probability))
    denominator = probability.sum(dim=-1, keepdim=True)
    return torch.where(any_active, probability / denominator.clamp_min(torch.finfo(torch.float32).tiny), probability)


def _native_atom_terms(
    query: torch.Tensor,
    native_key: torch.Tensor | None,
    native_value: torch.Tensor | None,
    native_valid: torch.Tensor | None,
    native_mask: torch.Tensor | None,
    working_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate optional ordinary native atoms and construct their attention terms."""

    supplied = (native_key is not None, native_value is not None, native_valid is not None)
    b, h, t, d = query.shape
    if not any(supplied):
        if native_mask is not None:
            raise ValueError("native_mask was supplied without native atoms")
        return (
            torch.empty(b, h, t, 0, device=query.device, dtype=working_dtype),
            torch.empty(b, h, t, 0, device=query.device, dtype=torch.bool),
            torch.empty(b, h, 0, d, device=query.device, dtype=query.dtype),
        )
    if not all(supplied):
        raise ValueError("native_key, native_value, and native_valid must be supplied together")
    assert native_key is not None and native_value is not None and native_valid is not None
    if (
        native_key.ndim != 4
        or native_value.shape != native_key.shape
        or native_key.shape[:2] != (b, h)
        or native_key.shape[-1] != d
    ):
        raise ValueError("native K/V shape mismatch")
    m = native_key.shape[2]
    if native_valid.shape != (b, m):
        raise ValueError("native_valid must have shape [B,M]")
    if native_valid.dtype != torch.bool:
        raise ValueError("native_valid must be bool")
    logits = torch.einsum(
        "bhtd,bhmd->bhtm", query.to(working_dtype), native_key.to(working_dtype)
    ) / math.sqrt(d)
    if native_mask is None:
        mask = torch.zeros(b, 1, t, m, device=query.device, dtype=working_dtype)
    else:
        mask = torch.broadcast_to(native_mask.to(working_dtype), (b, h, t, m))
    active = (
        native_valid[:, None, None, :]
        & torch.isfinite(mask)
        & (mask > torch.finfo(torch.float32).min / 2)
    )
    return logits + mask, active.expand(b, h, t, m), native_value


def centered_factorized_attention(
    query: torch.Tensor,
    candidate_key: torch.Tensor,
    candidate_value: torch.Tensor,
    prior: torch.Tensor,
    valid: torch.Tensor,
    lambda_value: float,
    parent_mask: torch.Tensor | None = None,
    *,
    native_key: torch.Tensor | None = None,
    native_value: torch.Tensor | None = None,
    native_valid: torch.Tensor | None = None,
    native_mask: torch.Tensor | None = None,
    return_native_probability: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-tensor global-softmax oracle for the centered candidate family.

    Shapes are Q=[B,H,T,D], candidate K/V=[B,H,N,K,D], A/valid=[B,N,K].
    Optional native atoms [B,H,M,D] are concatenated with all legal children
    before a single softmax. This transport oracle requires every transported
    parent to have support; production handles m=0 with its native fallback.
    """

    if query.ndim != 4 or candidate_key.ndim != 5 or candidate_value.shape != candidate_key.shape:
        raise ValueError("unexpected centered attention shapes")
    b, h, t, d = query.shape
    if candidate_key.shape[:2] != (b, h) or candidate_key.shape[-1] != d:
        raise ValueError("query and candidate K/V shapes are incompatible")
    n, k = candidate_key.shape[2:4]
    if prior.shape != valid.shape or prior.shape != (b, n, k):
        raise ValueError("prior shape mismatch")
    normalized, legal = canonical_prior(prior, valid)
    if bool((legal.sum(dim=-1) == 0).any()):
        raise ValueError("transport-only oracle requires support for every parent")
    key_lambda, _ = centered_candidates(candidate_key, normalized, legal, lambda_value)
    value_lambda, _ = centered_candidates(candidate_value, normalized, legal, lambda_value)
    working_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    raw = torch.einsum(
        "bhtd,bhnkd->bhtnk", query.to(working_dtype), key_lambda.to(working_dtype)
    ) / math.sqrt(d)
    log_prior = torch.where(
        legal,
        normalized.clamp_min(torch.finfo(torch.float32).tiny).log(),
        torch.full_like(normalized, -torch.inf),
    )
    logits = raw + log_prior[:, None, None, :, :]
    if parent_mask is None:
        mask = torch.zeros(b, h, t, n, device=query.device, dtype=working_dtype)
    else:
        mask = torch.broadcast_to(parent_mask.to(working_dtype), (b, h, t, n))
    logits = logits + mask.unsqueeze(-1)
    active = (
        legal[:, None, None, :, :]
        & torch.isfinite(mask.unsqueeze(-1))
        & (mask.unsqueeze(-1) > torch.finfo(torch.float32).min / 2)
    )
    child_logits = logits.flatten(-2)
    child_active = active.expand(b, h, t, n, k).flatten(-2)
    native_logits, native_active, native_values = _native_atom_terms(
        query, native_key, native_value, native_valid, native_mask, working_dtype
    )
    all_probability = _masked_softmax(
        torch.cat((native_logits, child_logits), dim=-1),
        torch.cat((native_active, child_active), dim=-1),
    )
    native_count = native_logits.shape[-1]
    native_probability = all_probability[..., :native_count]
    child_probability = all_probability[..., native_count:].reshape(b, h, t, n, k)
    output = torch.einsum(
        "bhtnk,bhnkd->bhtd", child_probability, value_lambda.to(working_dtype)
    )
    if native_count:
        output = output + torch.einsum(
            "bhtm,bhmd->bhtd", native_probability, native_values.to(working_dtype)
        )
    output = output.to(query.dtype)
    if return_native_probability:
        return output, child_probability, native_probability
    return output, child_probability


def collapsed_attention(
    query: torch.Tensor,
    candidate_key: torch.Tensor,
    candidate_value: torch.Tensor,
    prior: torch.Tensor,
    valid: torch.Tensor,
    parent_mask: torch.Tensor | None = None,
    *,
    native_key: torch.Tensor | None = None,
    native_value: torch.Tensor | None = None,
    native_valid: torch.Tensor | None = None,
    native_mask: torch.Tensor | None = None,
    return_native_probability: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """C_post oracle over prior-mean parent slots."""

    normalized, legal = canonical_prior(prior, valid)
    if bool((legal.sum(dim=-1) == 0).any()):
        raise ValueError("transport-only oracle requires support for every parent")
    weight = normalized[:, None, :, :, None].to(candidate_key.dtype)
    key = (candidate_key * weight).sum(dim=3)
    value = (candidate_value * weight).sum(dim=3)
    b, h, t, d = query.shape
    n = key.shape[2]
    working_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    logits = torch.einsum(
        "bhtd,bhnd->bhtn", query.to(working_dtype), key.to(working_dtype)
    ) / math.sqrt(d)
    if parent_mask is None:
        mask = torch.zeros(b, h, t, n, device=query.device, dtype=working_dtype)
    else:
        mask = torch.broadcast_to(parent_mask.to(working_dtype), (b, h, t, n))
    parent_logits = logits + mask
    parent_active = torch.isfinite(mask) & (
        mask > torch.finfo(torch.float32).min / 2
    )
    native_logits, native_active, native_values = _native_atom_terms(
        query, native_key, native_value, native_valid, native_mask, working_dtype
    )
    all_probability = _masked_softmax(
        torch.cat((native_logits, parent_logits), dim=-1),
        torch.cat((native_active, parent_active), dim=-1),
    )
    native_count = native_logits.shape[-1]
    native_probability = all_probability[..., :native_count]
    probability = all_probability[..., native_count:]
    output = torch.einsum(
        "bhtn,bhnd->bhtd", probability, value.to(working_dtype)
    )
    if native_count:
        output = output + torch.einsum(
            "bhtm,bhmd->bhtd", native_probability, native_values.to(working_dtype)
        )
    output = output.to(query.dtype)
    if return_native_probability:
        return output, probability, native_probability
    return output, probability


def causal_gold_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return teacher-forced log p(label[t+1] | prefix through t)."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected logits=[B,T,V] and labels=[B,T]")
    if logits.shape[1] < 2:
        empty = torch.empty(0, device=logits.device)
        return {
            "batch_index": empty.long(),
            "query_position": empty.long(),
            "target_position": empty.long(),
            "target_token_id": empty.long(),
            "gold_logp": empty,
        }
    shifted_labels = labels[:, 1:]
    eligible = shifted_labels != -100
    if bool(((shifted_labels[eligible] < 0) | (shifted_labels[eligible] >= logits.shape[-1])).any()):
        raise ValueError("eligible label is outside the vocabulary")
    batch, query = torch.where(eligible)
    target = query + 1
    target_id = shifted_labels[batch, query]
    log_probability = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    gold = log_probability[batch, query, target_id]
    return {
        "batch_index": batch,
        "query_position": query,
        "target_position": target,
        "target_token_id": target_id,
        "gold_logp": gold,
    }


def classify_candidate_topology(
    receiver_span: tuple[int, int],
    source_spans: Sequence[tuple[int, int]],
    *,
    candidate_origins: Sequence[str] | None = None,
    boundary_or_fallback: bool = False,
    independent_competitors: bool = False,
    certified_partition: bool = False,
    duplicate_or_overlap_alias: bool = False,
) -> str:
    """Conservatively classify raw m>=2 geometry.

    Merely failing the partition test is not evidence that candidates are
    independent alternatives.  That class requires an explicit upstream
    semantic/geometry determination; otherwise the row stays unresolved.
    """

    if len(source_spans) < 2:
        return "not_applicable"
    origins = list(candidate_origins or ("span_overlap",) * len(source_spans))
    if len(origins) != len(source_spans):
        raise ValueError("candidate origin count mismatch")
    start, end = receiver_span
    malformed = start < 0 or end <= start or any(
        a < 0 or b <= a for a, b in source_spans
    )
    if boundary_or_fallback or duplicate_or_overlap_alias or malformed:
        return "boundary_fallback"
    if any(origin == "window_neighbor" for origin in origins):
        return "neighbor_expansion"
    intersections = [
        (max(start, a), min(end, b))
        for a, b in source_spans
        if min(end, b) > max(start, a)
    ]
    if len(intersections) == len(source_spans):
        ordered_intersections = sorted(intersections)
        cursor = start
        partition = True
        for left, right in ordered_intersections:
            if left != cursor or right <= left:
                partition = False
                break
            cursor = right
        if partition and cursor == end and certified_partition:
            return "partition_compositional"
    elif certified_partition:
        raise ValueError("certified partition lacks one intersection per candidate")
    if certified_partition:
        raise ValueError("certified partition geometry is not a disjoint complete cover")
    complete_explanations = all(
        left == start and right == end for left, right in intersections
    )
    explicit_competition = independent_competitors or all(
        origin == "independent_complete_explanation" for origin in origins
    )
    if (
        explicit_competition
        and len(intersections) == len(source_spans)
        and complete_explanations
        and len(set(source_spans)) == len(source_spans)
    ):
        return "competing_overlap"
    return "taxonomy_unresolved"


def _validate_distribution(values: Any, count: int, name: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"{name} must contain candidate_count values")
    result = [_finite_float(value, f"{name}[{index}]", nonnegative=True) for index, value in enumerate(values)]
    if count and any(value <= 0 for value in result if name == "prior"):
        raise ValueError("canonical legal prior entries must be strictly positive")
    if count and abs(sum(result) - 1.0) > FLOAT32_ATOL:
        raise ValueError(f"{name} must sum to one")
    return result


def attach_stream_identity(
    row: Mapping[str, Any], row_ordinal: int
) -> dict[str, Any]:
    """Attach the A4 physical locator and matched endpoint identities.

    This utility does not assign ordinals; callers must use the frozen
    sample-local Cartesian formula.  It is used by synthetic fixtures and is
    also suitable for producer-side contract tests.
    """

    result = dict(row)
    result["row_ordinal"] = _integer(row_ordinal, "row_ordinal")
    if result["row_ordinal"] < 0:
        raise ValueError("row_ordinal must be nonnegative")
    result["logical_row_id"] = expected_logical_row_id(result)
    result["endpoint_id"] = canonical_endpoint_id(
        result["seed"],
        result["checkpoint_arm"],
        result["inference_operator"],
        result["cell"],
        result["task"],
        float(result["lambda_value"]),
    )
    result["endpoint_row_id"] = expected_endpoint_row_id(
        result["endpoint_id"], result["logical_row_id"]
    )
    return result


def validate_and_derive_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in INPUT_COLUMNS if name not in raw]
    if missing:
        raise ValueError(f"mechanism row missing columns: {missing}")
    row = {name: raw[name] for name in INPUT_COLUMNS}
    if _integer(row["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise ValueError("mechanism row schema version mismatch")
    if row["split_role"] != SPLIT_ROLE:
        raise ValueError("data firewall: only split_role=e0_design is permitted")
    row["seed"] = _integer(row["seed"], "seed")
    if row["checkpoint_arm"] not in CHECKPOINT_ARMS:
        raise ValueError("unknown checkpoint arm")
    if row["inference_operator"] not in INFERENCE_OPERATORS:
        raise ValueError("unknown inference operator")
    expected_cell = CELL_MAP[(row["checkpoint_arm"], row["inference_operator"])]
    if row["cell"] != expected_cell:
        raise ValueError(f"cell mismatch; expected {expected_cell}")
    if row["task"] not in TASKS:
        raise ValueError("unknown task")
    for name in (
        "sample_sha256",
        "content_group_sha256",
        "input_sha256",
        "alignment_sha256",
        "labels_sha256",
        "gold_response_sha256",
    ):
        row[name] = _sha256(row[name], name)
    for name in (
        "layer",
        "query_head",
        "kv_head",
        "query_position",
        "target_position",
        "target_token_id",
        "parent_position",
        "candidate_count",
        "row_ordinal",
    ):
        row[name] = _integer(row[name], name)
    if row["row_ordinal"] < 0:
        raise ValueError("row_ordinal must be nonnegative")
    if not 2 <= row["candidate_count"] <= 4:
        raise ValueError("functional mechanism rows require certified 2<=candidate_count<=4")
    if row["kv_head"] > row["query_head"]:
        raise ValueError("kv_head/query_head mapping is impossible")
    if row["target_position"] != row["query_position"] + 1:
        raise ValueError("teacher-forcing causal shift mismatch")
    if row["topology"] not in TOPOLOGIES:
        raise ValueError("unknown topology")
    if row["topology"] == "not_applicable":
        raise ValueError("functional mechanism rows cannot use not_applicable topology")
    row["lambda_value"] = _grid_value(row["lambda_value"])
    if row["inference_operator"] == "c_post" and row["lambda_value"] != 0.0:
        raise ValueError("C_post baseline is only represented at lambda=0")
    expected_endpoint = canonical_endpoint_id(
        row["seed"],
        row["checkpoint_arm"],
        row["inference_operator"],
        row["cell"],
        row["task"],
        row["lambda_value"],
    )
    if row["endpoint_id"] != expected_endpoint:
        raise ValueError("endpoint_id differs from the frozen endpoint identity")
    logical_id = expected_logical_row_id(row)
    if row["logical_row_id"] != logical_id:
        raise ValueError("logical_row_id differs from the frozen existing row key")
    endpoint_logical_id = expected_endpoint_row_id(expected_endpoint, logical_id)
    if row["endpoint_row_id"] != endpoint_logical_id:
        raise ValueError("endpoint_row_id differs from endpoint_id/logical_row_id")
    count = row["candidate_count"]
    row["prior"] = _validate_distribution(row["prior"], count, "prior")
    candidate_indices = row["candidate_indices"]
    candidate_valid = row["candidate_valid_mask"]
    candidate_slot_weights = row["candidate_slot_weights"]
    if not (
        isinstance(candidate_indices, list)
        and isinstance(candidate_valid, list)
        and isinstance(candidate_slot_weights, list)
        and len(candidate_indices) == len(candidate_valid) == len(candidate_slot_weights) == 4
    ):
        raise ValueError("candidate slot geometry must contain exactly four slots")
    row["candidate_indices"] = [
        _integer(value, "candidate_indices", nonnegative=False)
        for value in candidate_indices
    ]
    if any(not isinstance(value, bool) for value in candidate_valid):
        raise ValueError("candidate_valid_mask must contain bool values")
    row["candidate_valid_mask"] = list(candidate_valid)
    row["candidate_slot_weights"] = [
        _finite_float(value, "candidate_slot_weights", nonnegative=True)
        for value in candidate_slot_weights
    ]
    expected_valid = [
        index >= 0 and weight > 0
        for index, weight in zip(
            row["candidate_indices"], row["candidate_slot_weights"]
        )
    ]
    legal_indices = [
        index
        for index, valid in zip(row["candidate_indices"], expected_valid)
        if valid
    ]
    legal_weights = [
        weight
        for weight, valid in zip(row["candidate_slot_weights"], expected_valid)
        if valid
    ]
    if (
        row["candidate_valid_mask"] != expected_valid
        or len(legal_indices) != count
        or len(set(legal_indices)) != count
        or any(
            (not valid) and (index != -1 or weight != 0.0)
            for index, weight, valid in zip(
                row["candidate_indices"],
                row["candidate_slot_weights"],
                row["candidate_valid_mask"],
            )
        )
        or any(
            abs(left - right) > FLOAT32_ATOL
            for left, right in zip(legal_weights, row["prior"])
        )
    ):
        raise ValueError("candidate identity/mask/slot weights differ from legal prior")
    row["statistical_weight"] = _finite_float(
        row["statistical_weight"], "statistical_weight", nonnegative=True
    )
    if row["statistical_weight"] <= 0:
        raise ValueError("statistical_weight must be strictly positive")
    row["gamma"] = _validate_distribution(row["gamma"], count, "gamma")
    for name in (
        "source_d_k", "source_d_v", "source_energy_k", "source_energy_v",
        "fused_d_k", "fused_d_v", "fused_energy_k", "fused_energy_v",
        "candidate_logit_range", "candidate_logit_variance", "jensen_gap",
        "parent_attention_mass", "output_delta_l2",
    ):
        row[name] = _finite_float(row[name], name, nonnegative=True)
    if row["parent_attention_mass"] > 1.0 + FLOAT32_ATOL:
        raise ValueError("parent_attention_mass exceeds one")
    row["gold_logp"] = _finite_float(row["gold_logp"], "gold_logp")
    row["cpost_gold_logp"] = _finite_float(row["cpost_gold_logp"], "cpost_gold_logp")
    if not isinstance(row["end_task_correct"], bool) or not isinstance(row["cpost_end_task_correct"], bool):
        raise ValueError("end-task correctness fields must be bool")

    kl = 0.0
    tv = 0.0
    for posterior, prior in zip(row["gamma"], row["prior"]):
        if posterior > 0:
            if prior <= 0:
                raise ValueError("posterior assigns mass outside prior support")
            kl += posterior * math.log(posterior / prior)
        tv += abs(posterior - prior)
    row["gamma_kl_prior"] = max(0.0, kl)
    row["gamma_tv_prior"] = 0.5 * tv
    row["gamma_query_variance"] = 0.0  # Filled by cross-query Welford reduction.
    row["posterior_top1_candidate"] = (
        max(range(count), key=lambda index: (row["gamma"][index], -index)) if count else -1
    )
    row["posterior_top1_changed"] = False
    row["normalized_source_d_k"] = row["source_d_k"] / (row["source_energy_k"] + EPSILON)
    row["normalized_source_d_v"] = row["source_d_v"] / (row["source_energy_v"] + EPSILON)
    row["normalized_fused_d_k"] = row["fused_d_k"] / (row["fused_energy_k"] + EPSILON)
    row["normalized_fused_d_v"] = row["fused_d_v"] / (row["fused_energy_v"] + EPSILON)
    row["projector_retention_k"] = row["fused_d_k"] / (row["source_d_k"] + EPSILON)
    row["projector_retention_v"] = row["fused_d_v"] / (row["source_d_v"] + EPSILON)
    row["delta_gold_logp"] = row["gold_logp"] - row["cpost_gold_logp"]
    row["end_task_accuracy_flip"] = (
        row["end_task_correct"] != row["cpost_end_task_correct"]
    )
    if row["lambda_value"] == 0.0:
        gamma_prior_delta = max(
            (abs(gamma - prior) for gamma, prior in zip(row["gamma"], row["prior"])),
            default=0.0,
        )
        if (
            abs(row["delta_gold_logp"]) > FLOAT32_ATOL
            or row["output_delta_l2"] > FLOAT32_ATOL
            or row["candidate_logit_range"] > FLOAT32_ATOL
            or row["candidate_logit_variance"] > FLOAT32_ATOL
            or row["jensen_gap"] > FLOAT32_ATOL
            or gamma_prior_delta > FLOAT32_ATOL
            or row["end_task_accuracy_flip"]
        ):
            raise ValueError("lambda=0 violates exact C_post row oracle")
    return row


def _query_group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["seed"], row["checkpoint_arm"], row["inference_operator"], row["task"],
        row["sample_sha256"], row["layer"], row["query_head"], row["parent_position"],
        row["lambda_value"],
    )


def _finish_prepared_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill cross-query fields and deterministically sort one bounded row set."""

    if not rows:
        raise ValueError("mechanism audit input is empty")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_query_group_key(row)].append(row)
    for members in grouped.values():
        count = members[0]["candidate_count"]
        if count == 0:
            continue
        seen_queries: set[int] = set()
        accumulator = GammaQueryAccumulator(count)
        for row in sorted(members, key=lambda value: value["query_position"]):
            query_position = row["query_position"]
            if query_position in seen_queries:
                raise ValueError("duplicate query-parent-layer-head mechanism row")
            seen_queries.add(query_position)
            accumulator.update(row["gamma"])
        for row in members:
            row["gamma_query_variance"] = accumulator.mean_candidate_variance
            row["posterior_top1_changed"] = accumulator.top1_changed
    rows.sort(key=lambda row: tuple(row[name] for name in FUNCTIONAL_SORT_COLUMNS))
    return rows


def prepare_rows(raw_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Small synthetic reference; never part of the formal analyzer path."""

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if len(rows) >= MAX_SYNTHETIC_REFERENCE_ROWS:
            raise ValueError("synthetic in-memory reference exceeds its test-only bound")
        rows.append(validate_and_derive_row(raw))
    return _finish_prepared_rows(rows)


def _mean(rows: Sequence[Mapping[str, Any]], name: str) -> float:
    return sum(float(row[name]) for row in rows) / len(rows)


def _group_rows(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        record = {name: value for name, value in zip(keys, key)}
        record["row_count"] = len(members)
        record["sample_count"] = len({row["sample_sha256"] for row in members})
        record["content_group_count"] = len({row["content_group_sha256"] for row in members})
        record["unique_parent_count"] = len(
            {
                (
                    row["sample_sha256"],
                    row["layer"],
                    row["parent_position"],
                )
                for row in members
            }
        )
        for name in MEAN_COLUMNS:
            record[f"mean_{name}"] = _mean(members, name)
        group_flip: dict[str, bool] = {}
        for row in members:
            previous = group_flip.setdefault(
                row["content_group_sha256"], row["end_task_accuracy_flip"]
            )
            if previous != row["end_task_accuracy_flip"]:
                raise ValueError("end-task correctness flip differs within content group")
        record["group_equal_end_task_accuracy_flip_rate"] = (
            sum(group_flip.values()) / len(group_flip)
        )
        record["posterior_top1_change_rate"] = _mean(members, "posterior_top1_changed")
        output.append(record)
    return output


def layer_head_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return _group_rows(
        rows,
        (
            "seed", "checkpoint_arm", "inference_operator", "cell", "task",
            "lambda_value", "layer", "query_head", "kv_head",
        ),
    )


def _consistent_parent_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate source/fused geometry repeated across query/operator/lambda."""

    keys = (
        "seed", "checkpoint_arm", "task", "sample_sha256", "content_group_sha256",
        "layer", "parent_position",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, members in sorted(grouped.items()):
        reference = members[0]
        for other in members[1:]:
            for name in PARENT_GEOMETRY_METRICS:
                if isinstance(reference[name], str):
                    equal = reference[name] == other[name]
                else:
                    equal = math.isclose(float(reference[name]), float(other[name]), abs_tol=FLOAT32_ATOL, rel_tol=2e-5)
                if not equal:
                    raise ValueError(f"pre-collapse candidate geometry differs across operator/query/lambda: {name}")
        output.append(
            {
                **{name: value for name, value in zip(keys, key)},
                **{name: reference[name] for name in PARENT_GEOMETRY_METRICS},
            }
        )
    return output


def contraction_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    parents = _consistent_parent_rows(rows)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ("seed", "checkpoint_arm", "task", "layer", "topology")
    for row in parents:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, members in sorted(groups.items()):
        record = {name: value for name, value in zip(keys, key)}
        record["parent_count"] = len(members)
        record["content_group_count"] = len({row["content_group_sha256"] for row in members})
        for name in (
            "source_d_k", "source_d_v", "fused_d_k", "fused_d_v",
            "normalized_source_d_k", "normalized_source_d_v", "normalized_fused_d_k",
            "normalized_fused_d_v", "projector_retention_k", "projector_retention_v",
        ):
            record[f"mean_{name}"] = _mean(members, name)
        output.append(record)
    return output


def topology_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return _group_rows(
        rows,
        ("seed", "checkpoint_arm", "inference_operator", "cell", "task", "lambda_value", "topology"),
    )


def lambda_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return _group_rows(rows, ("seed", "checkpoint_arm", "inference_operator", "cell", "task", "lambda_value"))


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot write empty CSV")
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _unique_query_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "seed", "checkpoint_arm", "inference_operator", "cell", "task", "sample_sha256",
        "content_group_sha256", "query_position", "target_position", "lambda_value",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in keys)].append(row)
    unique = []
    for key, members in sorted(groups.items()):
        reference = members[0]
        for other in members[1:]:
            for name in (
                "gold_logp",
                "cpost_gold_logp",
                "end_task_correct",
                "cpost_end_task_correct",
                "target_token_id",
                "input_sha256",
                "alignment_sha256",
                "labels_sha256",
                "gold_response_sha256",
            ):
                if reference[name] != other[name] and not (
                    isinstance(reference[name], float)
                    and math.isclose(reference[name], other[name], abs_tol=FLOAT32_ATOL, rel_tol=2e-5)
                ):
                    raise ValueError(f"query-level outcome differs across layer/head/parent: {name}")
        unique.append({
            **{name: value for name, value in zip(keys, key)},
            "input_sha256": reference["input_sha256"],
            "alignment_sha256": reference["alignment_sha256"],
            "labels_sha256": reference["labels_sha256"],
            "gold_response_sha256": reference["gold_response_sha256"],
            "target_token_id": reference["target_token_id"],
            "gold_logp": reference["gold_logp"],
            "cpost_gold_logp": reference["cpost_gold_logp"],
            "delta_gold_logp": reference["delta_gold_logp"],
            "end_task_correct": reference["end_task_correct"],
            "cpost_end_task_correct": reference["cpost_end_task_correct"],
            "end_task_accuracy_flip": reference["end_task_accuracy_flip"],
        })
    return unique


def _group_equal_response_summary(
    query_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sum token log-probability within response, then weight content groups equally."""

    sample_keys = (
        "seed", "checkpoint_arm", "inference_operator", "cell", "task",
        "sample_sha256", "content_group_sha256", "lambda_value",
    )
    by_sample: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        by_sample[tuple(row[name] for name in sample_keys)].append(row)

    sample_rows: list[dict[str, Any]] = []
    for key, members in sorted(by_sample.items()):
        ordered = sorted(members, key=lambda row: row["query_position"])
        if len({row["query_position"] for row in ordered}) != len(ordered):
            raise ValueError("duplicate answer query in response summary")
        reference = ordered[0]
        for other in ordered[1:]:
            for name in (
                "end_task_correct",
                "cpost_end_task_correct",
                "input_sha256",
                "alignment_sha256",
                "labels_sha256",
                "gold_response_sha256",
            ):
                if other[name] != reference[name]:
                    raise ValueError(f"response-level field differs across queries: {name}")
        sample_rows.append(
            {
                **{name: value for name, value in zip(sample_keys, key)},
                "answer_query_count": len(ordered),
                "response_gold_logp": sum(row["gold_logp"] for row in ordered),
                "response_cpost_gold_logp": sum(
                    row["cpost_gold_logp"] for row in ordered
                ),
                "response_delta_gold_logp": sum(
                    row["delta_gold_logp"] for row in ordered
                ),
                "end_task_correct": reference["end_task_correct"],
                "cpost_end_task_correct": reference["cpost_end_task_correct"],
                "end_task_accuracy_flip": reference["end_task_accuracy_flip"],
            }
        )

    group_keys = (
        "seed", "checkpoint_arm", "inference_operator", "cell", "task",
        "content_group_sha256", "lambda_value",
    )
    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        by_group[tuple(row[name] for name in group_keys)].append(row)
    group_rows: list[dict[str, Any]] = []
    for key, members in sorted(by_group.items()):
        reference = members[0]
        for other in members[1:]:
            if (
                other["end_task_correct"] != reference["end_task_correct"]
                or other["cpost_end_task_correct"]
                != reference["cpost_end_task_correct"]
            ):
                raise ValueError("end-task correctness differs within content group")
        group_rows.append(
            {
                **{name: value for name, value in zip(group_keys, key)},
                "sample_count": len(members),
                "answer_query_count": sum(row["answer_query_count"] for row in members),
                "response_gold_logp": _mean(members, "response_gold_logp"),
                "response_cpost_gold_logp": _mean(
                    members, "response_cpost_gold_logp"
                ),
                "response_delta_gold_logp": _mean(
                    members, "response_delta_gold_logp"
                ),
                "end_task_correct": reference["end_task_correct"],
                "cpost_end_task_correct": reference["cpost_end_task_correct"],
                "end_task_accuracy_flip": reference["end_task_accuracy_flip"],
            }
        )
    return group_rows


def _build_summary_from_group_rows(
    *,
    row_count: int,
    answer_query_count: int,
    sample_ids: set[str],
    content_group_ids: set[str],
    seeds: set[int],
    cells: set[str],
    group_rows: Sequence[dict[str, Any]],
    artifact_sha256: Mapping[str, str],
) -> dict[str, Any]:
    primary_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    primary_keys = ("seed", "checkpoint_arm", "inference_operator", "cell", "task", "lambda_value")
    seen_group_keys: set[tuple[Any, ...]] = set()
    for row in sorted(
        group_rows,
        key=lambda value: (
            *(value[name] for name in primary_keys),
            value["content_group_sha256"],
        ),
    ):
        exact_group_key = (
            *(row[name] for name in primary_keys),
            row["content_group_sha256"],
        )
        if exact_group_key in seen_group_keys:
            raise ValueError(
                "streaming summary requires one canonical sample per content group"
            )
        seen_group_keys.add(exact_group_key)
        primary_groups[tuple(row[name] for name in primary_keys)].append(row)
    primary = []
    for key, members in sorted(primary_groups.items()):
        record = {name: value for name, value in zip(primary_keys, key)}
        record.update(
            answer_query_count=sum(row["answer_query_count"] for row in members),
            sample_count=sum(row["sample_count"] for row in members),
            content_group_count=len(members),
            group_equal_mean_response_gold_logp=_mean(
                members, "response_gold_logp"
            ),
            group_equal_mean_response_cpost_gold_logp=_mean(
                members, "response_cpost_gold_logp"
            ),
            group_equal_mean_response_delta_gold_logp=_mean(
                members, "response_delta_gold_logp"
            ),
            group_equal_end_task_accuracy=_mean(members, "end_task_correct"),
            group_equal_cpost_end_task_accuracy=_mean(
                members, "cpost_end_task_correct"
            ),
            group_equal_end_task_accuracy_flip_rate=_mean(
                members, "end_task_accuracy_flip"
            ),
        )
        primary.append(record)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "split_role": SPLIT_ROLE,
        "lambda_grid": list(LAMBDA_GRID),
        "row_count": row_count,
        "answer_query_count": answer_query_count,
        "sample_count": len(sample_ids),
        "content_group_count": len(content_group_ids),
        "seeds": sorted(seeds),
        "cells": sorted(cells),
        "primary_teacher_forced_gold_logp": primary,
        "artifact_sha256": dict(sorted(artifact_sha256.items())),
        "integrity": {
            "only_e0_design_consumed": True,
            "e1_pilot_consumed": False,
            "confirmatory_consumed": False,
            "causal_shift_verified": True,
            "lambda_zero_exact_cpost_rows": True,
            "precollapse_geometry_operator_invariant": True,
            "representation_preserving_streaming": True,
            "formal_reduction_order": "endpoint_id/sample_sha256/row_ordinal",
            "logical_row_ceiling": None,
        },
        "claim_boundary": "E0-design mechanism localization only; no E1-pilot or confirmatory performance claim.",
    }


def build_summary(
    rows: Sequence[dict[str, Any]], artifact_sha256: Mapping[str, str]
) -> dict[str, Any]:
    query_rows = _unique_query_summary(rows)
    return _build_summary_from_group_rows(
        row_count=len(rows),
        answer_query_count=len(query_rows),
        sample_ids={row["sample_sha256"] for row in rows},
        content_group_ids={row["content_group_sha256"] for row in rows},
        seeds={row["seed"] for row in rows},
        cells={row["cell"] for row in rows},
        group_rows=_group_equal_response_summary(query_rows),
        artifact_sha256=artifact_sha256,
    )


class _GroupedRecordCombiner:
    """Combine bounded per-sample `_group_rows` outputs without retaining rows."""

    def __init__(self, keys: Sequence[str]) -> None:
        self.keys = tuple(keys)
        self.states: dict[tuple[Any, ...], dict[str, Any]] = {}

    def add_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        local_parents: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
        local_flip: dict[tuple[Any, ...], bool] = {}
        for row in rows:
            key = tuple(row[name] for name in self.keys)
            state = self.states.setdefault(
                key,
                {
                    "row_count": 0,
                    "sample_count": 0,
                    "content_group_count": 0,
                    "unique_parent_count": 0,
                    "mean_sums": {name: 0.0 for name in MEAN_COLUMNS},
                    "group_flip_sum": 0,
                    "posterior_top1_changed_sum": 0.0,
                },
            )
            state["row_count"] += 1
            for name in MEAN_COLUMNS:
                state["mean_sums"][name] += float(row[name])
            flip = bool(row["end_task_accuracy_flip"])
            previous = local_flip.setdefault(key, flip)
            if previous != flip:
                raise ValueError("end-task correctness flip differs within content group")
            state["posterior_top1_changed_sum"] += float(
                row["posterior_top1_changed"]
            )
            local_parents[key].add(
                (
                    row["sample_sha256"],
                    row["layer"],
                    row["parent_position"],
                )
            )
        for key, parents in local_parents.items():
            self.states[key]["unique_parent_count"] += len(parents)
            self.states[key]["sample_count"] += 1
            self.states[key]["content_group_count"] += 1
            self.states[key]["group_flip_sum"] += int(local_flip[key])

    def finish(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for key, state in sorted(self.states.items()):
            count = int(state["row_count"])
            record = {name: value for name, value in zip(self.keys, key)}
            record["row_count"] = count
            record["sample_count"] = int(state["sample_count"])
            record["content_group_count"] = int(state["content_group_count"])
            record["unique_parent_count"] = int(state["unique_parent_count"])
            for name in MEAN_COLUMNS:
                record[f"mean_{name}"] = state["mean_sums"][name] / count
            record["group_equal_end_task_accuracy_flip_rate"] = (
                state["group_flip_sum"] / state["content_group_count"]
            )
            record["posterior_top1_change_rate"] = (
                state["posterior_top1_changed_sum"] / count
            )
            output.append(record)
        return output


class _DiskBackedGroupedReducer:
    """Ordered floating reduction with disk-backed distinct ledgers.

    Floating sums are updated only by the caller's frozen logical-row order.
    SQLite is used solely for exact distinct membership and group-consistency
    checks, so neither physical chunk boundaries nor shard completion order can
    change a formal floating result.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        name: str,
        keys: Sequence[str],
    ) -> None:
        self.connection = connection
        self.name = name
        self.keys = tuple(keys)
        self.states: dict[str, dict[str, Any]] = {}

    def add(self, row: Mapping[str, Any]) -> None:
        key_values = [row[name] for name in self.keys]
        key = canonical_json_bytes(key_values).decode("utf-8").rstrip("\n")
        state = self.states.setdefault(
            key,
            {
                "key_values": key_values,
                "row_count": 0,
                "mean_sums": {name: 0.0 for name in MEAN_COLUMNS},
                "posterior_top1_changed_sum": 0.0,
                "sample_count": 0,
                "unique_parent_count": 0,
                "current_sample": None,
                "current_content_group": None,
                "current_first_query": None,
                "current_flip": None,
            },
        )
        sample = str(row["sample_sha256"])
        group = str(row["content_group_sha256"])
        flip = int(bool(row["end_task_accuracy_flip"]))
        if sample != state["current_sample"]:
            if state["current_sample"] is not None and sample < state["current_sample"]:
                raise ValueError("formal sample reduction order regressed")
            state["current_sample"] = sample
            state["current_content_group"] = group
            state["current_first_query"] = int(row["query_position"])
            state["current_flip"] = flip
            state["sample_count"] += 1
            cursor = self.connection.execute(
                """INSERT OR IGNORE INTO aggregate_group_flip(
                       reducer, aggregate_key, content_group_sha256, flip
                   ) VALUES (?, ?, ?, ?)""",
                (self.name, key, group, flip),
            )
            if cursor.rowcount == 0:
                stored = self.connection.execute(
                    """SELECT flip FROM aggregate_group_flip
                       WHERE reducer = ? AND aggregate_key = ?
                             AND content_group_sha256 = ?""",
                    (self.name, key, group),
                ).fetchone()
                if stored is None or int(stored[0]) != flip:
                    raise ValueError(
                        "end-task correctness flip differs within content group"
                    )
        elif (
            group != state["current_content_group"]
            or flip != state["current_flip"]
        ):
            raise ValueError("sample-level group/correctness fields changed")

        state["row_count"] += 1
        for metric in MEAN_COLUMNS:
            state["mean_sums"][metric] += float(row[metric])
        state["posterior_top1_changed_sum"] += float(
            row["posterior_top1_changed"]
        )
        first_query = int(state["current_first_query"])
        if "query_head" in self.keys:
            parent_representative = int(row["query_position"]) == first_query
        else:
            parent_representative = (
                int(row["query_head"]) == 0
                and int(row["query_position"]) == first_query
            )
        if parent_representative:
            state["unique_parent_count"] += 1

    def finish(self) -> list[dict[str, Any]]:
        groups: dict[str, tuple[int, int]] = {
            str(key): (int(count), int(flip_sum))
            for key, count, flip_sum in self.connection.execute(
                """SELECT aggregate_key, COUNT(*), SUM(flip)
                   FROM aggregate_group_flip WHERE reducer = ?
                   GROUP BY aggregate_key ORDER BY aggregate_key""",
                (self.name,),
            )
        }
        output: list[dict[str, Any]] = []
        ordered_states = sorted(
            self.states.values(), key=lambda state: tuple(state["key_values"])
        )
        for state in ordered_states:
            key_values = state["key_values"]
            key = canonical_json_bytes(key_values).decode("utf-8").rstrip("\n")
            count = int(state["row_count"])
            group_count, flip_sum = groups[key]
            record = {
                name: value for name, value in zip(self.keys, key_values)
            }
            record.update(
                row_count=count,
                sample_count=int(state["sample_count"]),
                content_group_count=group_count,
                unique_parent_count=int(state["unique_parent_count"]),
            )
            for metric in MEAN_COLUMNS:
                record[f"mean_{metric}"] = state["mean_sums"][metric] / count
            record["group_equal_end_task_accuracy_flip_rate"] = (
                flip_sum / group_count
            )
            record["posterior_top1_change_rate"] = (
                state["posterior_top1_changed_sum"] / count
            )
            output.append(record)
        return output


class _ContractionCombiner:
    def __init__(self) -> None:
        self.metric_names = (
            "source_d_k",
            "source_d_v",
            "fused_d_k",
            "fused_d_v",
            "normalized_source_d_k",
            "normalized_source_d_v",
            "normalized_fused_d_k",
            "normalized_fused_d_v",
            "projector_retention_k",
            "projector_retention_v",
        )
        self.keys = ("seed", "checkpoint_arm", "task", "layer", "topology")
        self.states: dict[tuple[Any, ...], dict[str, Any]] = {}

    def add(self, parent: Mapping[str, Any]) -> None:
        key = tuple(parent[name] for name in self.keys)
        state = self.states.setdefault(
            key,
            {
                "parent_count": 0,
                "content_group_ids": set(),
                "sums": {name: 0.0 for name in self.metric_names},
            },
        )
        state["parent_count"] += 1
        state["content_group_ids"].add(parent["content_group_sha256"])
        for name in self.metric_names:
            state["sums"][name] += float(parent[name])

    def finish(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for key, state in sorted(self.states.items()):
            count = int(state["parent_count"])
            record = {name: value for name, value in zip(self.keys, key)}
            record["parent_count"] = count
            record["content_group_count"] = len(state["content_group_ids"])
            for name in self.metric_names:
                record[f"mean_{name}"] = state["sums"][name] / count
            output.append(record)
        return output


def _register_parent_geometry(
    connection: sqlite3.Connection,
    parent: Mapping[str, Any],
    reference_order: tuple[str, float],
) -> None:
    key_columns = (
        "seed",
        "checkpoint_arm",
        "task",
        "sample_sha256",
        "content_group_sha256",
        "layer",
        "parent_position",
    )
    key = canonical_json_bytes([parent[name] for name in key_columns])
    geometry = {name: parent[name] for name in PARENT_GEOMETRY_METRICS}
    payload = canonical_json_bytes(geometry).decode("utf-8")
    parent_record = canonical_json_bytes(dict(parent)).decode("utf-8")
    order_payload = canonical_json_bytes(list(reference_order)).decode("utf-8")
    cursor = connection.execute(
        """INSERT OR IGNORE INTO parent_geometry(
               parent_key, seed, checkpoint_arm, task, sample_sha256,
               content_group_sha256, layer, parent_position,
               payload, parent_record, reference_order
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            key,
            parent["seed"],
            parent["checkpoint_arm"],
            parent["task"],
            parent["sample_sha256"],
            parent["content_group_sha256"],
            parent["layer"],
            parent["parent_position"],
            payload,
            parent_record,
            order_payload,
        ),
    )
    if cursor.rowcount == 1:
        return
    stored = connection.execute(
        "SELECT payload, reference_order FROM parent_geometry WHERE parent_key = ?",
        (key,),
    ).fetchone()
    if stored is None:
        raise RuntimeError("parent geometry index lost an existing key")
    reference = json.loads(stored[0])
    for name in PARENT_GEOMETRY_METRICS:
        if isinstance(reference[name], str):
            equal = reference[name] == geometry[name]
        else:
            equal = math.isclose(
                float(reference[name]),
                float(geometry[name]),
                abs_tol=FLOAT32_ATOL,
                rel_tol=2e-5,
            )
        if not equal:
            raise ValueError(
                f"pre-collapse candidate geometry differs across operator/query/lambda: {name}"
            )
    if reference_order < tuple(json.loads(stored[1])):
        connection.execute(
            """UPDATE parent_geometry
               SET payload = ?, parent_record = ?, reference_order = ?
               WHERE parent_key = ?""",
            (payload, parent_record, order_payload, key),
        )


def _fragment_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        previous: tuple[Any, ...] | None = None
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = tuple(row[name] for name in FUNCTIONAL_SORT_COLUMNS)
            if previous is not None and key <= previous:
                raise ValueError("functional partition fragment is not strictly sorted")
            previous = key
            yield row


def _merge_partition_fragments(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    iterators = [iter(_fragment_rows(path)) for path in sorted(paths)]
    heap: list[tuple[tuple[Any, ...], int, dict[str, Any]]] = []
    for index, iterator in enumerate(iterators):
        row = next(iterator, None)
        if row is not None:
            heapq.heappush(
                heap,
                (tuple(row[name] for name in FUNCTIONAL_SORT_COLUMNS), index, row),
            )
    previous: tuple[Any, ...] | None = None
    while heap:
        key, index, row = heapq.heappop(heap)
        if previous is not None and key <= previous:
            raise ValueError("functional detail row key is duplicate or unsorted")
        previous = key
        yield row
        following = next(iterators[index], None)
        if following is not None:
            heapq.heappush(
                heap,
                (
                    tuple(following[name] for name in FUNCTIONAL_SORT_COLUMNS),
                    index,
                    following,
                ),
            )


def _write_bounded_parquet(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    writer = None
    schema = None
    buffer: list[dict[str, Any]] = []
    row_count = 0

    def flush() -> None:
        nonlocal writer, schema
        if not buffer:
            return
        records = [{name: row[name] for name in OUTPUT_COLUMNS} for row in buffer]
        table = pa.Table.from_pylist(records, schema=schema)
        if schema is None:
            table = table.select(list(OUTPUT_COLUMNS))
            schema = table.schema
            writer = pq.ParquetWriter(
                temporary,
                schema,
                compression="zstd",
                use_dictionary=False,
                write_statistics=True,
            )
        assert writer is not None
        writer.write_table(table)
        buffer.clear()

    try:
        for row in rows:
            buffer.append(dict(row))
            row_count += 1
            if len(buffer) >= PARQUET_BATCH_ROWS:
                flush()
        flush()
        if writer is None:
            raise ValueError("mechanism audit input is empty")
        writer.close()
        writer = None
        os.replace(temporary, path)
    finally:
        if writer is not None:
            writer.close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    return row_count


def _query_group_identity(row: Mapping[str, Any]) -> str:
    return canonical_json_bytes(
        [
            row["endpoint_id"],
            row["sample_sha256"],
            row["layer"],
            row["query_head"],
            row["parent_position"],
        ]
    ).decode("utf-8").rstrip("\n")


def _parent_geometry_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "seed",
        "checkpoint_arm",
        "task",
        "sample_sha256",
        "content_group_sha256",
        "layer",
        "parent_position",
    )
    return {
        **{name: row[name] for name in keys},
        **{name: row[name] for name in PARENT_GEOMETRY_METRICS},
    }


def _stage_functional_rows(
    connection: sqlite3.Connection,
    raw_rows: Iterable[Mapping[str, Any]],
) -> tuple[int, set[str], set[str], set[int], set[str], set[tuple[Any, ...]]]:
    """Pass one: validate and persist rows without retaining a sample table."""

    row_count = 0
    sample_ids: set[str] = set()
    content_group_ids: set[str] = set()
    seeds: set[int] = set()
    cells: set[str] = set()
    partitions: set[tuple[Any, ...]] = set()
    for raw in raw_rows:
        row = validate_and_derive_row(raw)
        partition = tuple(row[name] for name in PARTITION_COLUMNS)
        partitions.add(partition)
        if len(partitions) > MAX_STAGE_PARTITIONS:
            raise ValueError("functional stage exceeds frozen partition count")
        sample_ids.add(str(row["sample_sha256"]))
        content_group_ids.add(str(row["content_group_sha256"]))
        if (
            len(sample_ids) > MAX_E0_DESIGN_SAMPLES
            or len(content_group_ids) > MAX_E0_DESIGN_SAMPLES
        ):
            raise ValueError("functional stage exceeds frozen E0-design population")
        seeds.add(int(row["seed"]))
        cells.add(str(row["cell"]))
        payload = canonical_json_bytes(
            {name: row[name] for name in OUTPUT_COLUMNS}
        ).decode("utf-8").rstrip("\n")
        try:
            connection.execute(
                """INSERT INTO staged_rows(
                       endpoint_row_id, endpoint_id, task, sample_sha256,
                       content_group_sha256, row_ordinal, logical_row_id,
                       layer, query_head, query_position, parent_position,
                       query_group, payload
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["endpoint_row_id"],
                    row["endpoint_id"],
                    row["task"],
                    row["sample_sha256"],
                    row["content_group_sha256"],
                    row["row_ordinal"],
                    row["logical_row_id"],
                    row["layer"],
                    row["query_head"],
                    row["query_position"],
                    row["parent_position"],
                    _query_group_identity(row),
                    payload,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                "duplicate endpoint row or endpoint/sample row_ordinal"
            ) from error
        row_count += 1
    if row_count == 0:
        raise ValueError("mechanism audit input is empty")
    connection.commit()
    return row_count, sample_ids, content_group_ids, seeds, cells, partitions


def _first_occurrence_values(
    connection: sqlite3.Connection,
    endpoint_id: str,
    sample_sha256: str,
    column: str,
) -> list[int]:
    if column not in {"layer", "query_head", "query_position", "parent_position"}:
        raise ValueError("unsupported ordinal coordinate")
    return [
        int(value)
        for value, _ in connection.execute(
            f"""SELECT {column}, MIN(row_ordinal) FROM staged_rows
                WHERE endpoint_id = ? AND sample_sha256 = ?
                GROUP BY {column} ORDER BY MIN(row_ordinal), {column}""",
            (endpoint_id, sample_sha256),
        )
    ]


def _validate_staged_cartesian(connection: sqlite3.Connection) -> None:
    """Prove each endpoint/sample ordinal is a complete Cartesian bijection."""

    connection.execute(
        """CREATE TABLE logical_universe (
               task TEXT NOT NULL,
               sample_sha256 TEXT NOT NULL,
               row_ordinal INTEGER NOT NULL,
               logical_row_id TEXT NOT NULL,
               PRIMARY KEY(task, sample_sha256, row_ordinal)
           )"""
    )
    connection.execute(
        """INSERT OR IGNORE INTO logical_universe(
               task, sample_sha256, row_ordinal, logical_row_id
           )
           SELECT task, sample_sha256, row_ordinal, logical_row_id
           FROM staged_rows
           ORDER BY endpoint_id, sample_sha256, row_ordinal"""
    )
    mismatch = connection.execute(
        """SELECT 1 FROM staged_rows AS staged
           JOIN logical_universe AS frozen
             ON frozen.task = staged.task
            AND frozen.sample_sha256 = staged.sample_sha256
            AND frozen.row_ordinal = staged.row_ordinal
           WHERE frozen.logical_row_id != staged.logical_row_id
           LIMIT 1"""
    ).fetchone()
    if mismatch is not None:
        raise ValueError("logical row identity differs across matched endpoints")
    groups = connection.execute(
        """SELECT endpoint_id, task, sample_sha256, COUNT(*),
                  MIN(row_ordinal), MAX(row_ordinal),
                  COUNT(DISTINCT content_group_sha256)
           FROM staged_rows
           GROUP BY endpoint_id, task, sample_sha256
           ORDER BY endpoint_id, sample_sha256"""
    )
    for endpoint, task, sample, count, minimum, maximum, group_count in groups:
        count = int(count)
        if int(minimum) != 0 or int(maximum) != count - 1:
            raise ValueError("row_ordinal range is not contiguous from zero")
        if int(group_count) != 1:
            raise ValueError("content-group identity changes within endpoint/sample")
        layers = _first_occurrence_values(connection, endpoint, sample, "layer")
        heads = _first_occurrence_values(
            connection, endpoint, sample, "query_head"
        )
        queries = _first_occurrence_values(
            connection, endpoint, sample, "query_position"
        )
        parents = _first_occurrence_values(
            connection, endpoint, sample, "parent_position"
        )
        if not all((layers, heads, queries, parents)):
            raise ValueError("ordinal geometry has an empty Cartesian axis")
        if layers != list(range(len(layers))) or heads != list(range(len(heads))):
            raise ValueError("layer/query-head coordinates are not zero-based dense")
        if queries != sorted(queries) or parents != sorted(parents):
            raise ValueError(
                "row_ordinal does not encode the frozen Cartesian order"
            )
        expected_count = len(layers) * len(heads) * len(queries) * len(parents)
        if count != expected_count:
            raise ValueError(
                "endpoint/sample rows are not a complete logical Cartesian universe"
            )
        parent_count = len(parents)
        query_count = len(queries)
        head_count = len(heads)
        cursor = connection.execute(
            """SELECT row_ordinal, layer, query_head, query_position,
                      parent_position
               FROM staged_rows
               WHERE endpoint_id = ? AND sample_sha256 = ?
               ORDER BY row_ordinal""",
            (endpoint, sample),
        )
        for expected_ordinal, values in enumerate(cursor):
            ordinal, layer, head, query, parent = values
            if int(ordinal) != expected_ordinal:
                raise ValueError("row_ordinal contains a gap or reorder")
            parent_index = expected_ordinal % parent_count
            value = expected_ordinal // parent_count
            query_index = value % query_count
            value //= query_count
            head_index = value % head_count
            layer_index = value // head_count
            if (
                int(layer) != layers[layer_index]
                or int(head) != heads[head_index]
                or int(query) != queries[query_index]
                or int(parent) != parents[parent_index]
            ):
                raise ValueError("row_ordinal does not encode the frozen Cartesian order")
    connection.commit()


def _prepare_query_statistics(connection: sqlite3.Connection) -> None:
    """Pass-two prelude: reduce gamma in canonical group/query order."""

    current_key: str | None = None
    accumulator: GammaQueryAccumulator | None = None
    seen_queries: set[int] = set()

    def flush() -> None:
        if current_key is None or accumulator is None:
            return
        connection.execute(
            "INSERT INTO query_statistics VALUES (?, ?, ?)",
            (
                current_key,
                accumulator.mean_candidate_variance,
                int(accumulator.top1_changed),
            ),
        )

    cursor = connection.execute(
        """SELECT query_group, query_position, payload FROM staged_rows
           ORDER BY query_group, query_position, row_ordinal"""
    )
    for key, query_position, payload in cursor:
        row = json.loads(payload)
        if key != current_key:
            flush()
            current_key = str(key)
            accumulator = GammaQueryAccumulator(int(row["candidate_count"]))
            seen_queries = set()
        assert accumulator is not None
        query_position = int(query_position)
        if query_position in seen_queries:
            raise ValueError("duplicate query-parent-layer-head mechanism row")
        seen_queries.add(query_position)
        accumulator.update(row["gamma"])
    flush()
    connection.commit()


_QUERY_OUTCOME_FIELDS = (
    "seed",
    "checkpoint_arm",
    "inference_operator",
    "cell",
    "task",
    "sample_sha256",
    "content_group_sha256",
    "query_position",
    "target_position",
    "lambda_value",
    "input_sha256",
    "alignment_sha256",
    "labels_sha256",
    "gold_response_sha256",
    "target_token_id",
    "gold_logp",
    "cpost_gold_logp",
    "delta_gold_logp",
    "end_task_correct",
    "cpost_end_task_correct",
    "end_task_accuracy_flip",
)


def _register_query_outcome(
    connection: sqlite3.Connection, row: Mapping[str, Any]
) -> None:
    key = canonical_json_bytes(
        [
            row["endpoint_id"],
            row["sample_sha256"],
            row["query_position"],
            row["target_position"],
        ]
    ).decode("utf-8").rstrip("\n")
    payload = canonical_json_bytes(
        {name: row[name] for name in _QUERY_OUTCOME_FIELDS}
    ).decode("utf-8").rstrip("\n")
    cursor = connection.execute(
        """INSERT OR IGNORE INTO query_outcomes(
               query_key, endpoint_id, sample_sha256, query_position, payload
           ) VALUES (?, ?, ?, ?, ?)""",
        (key, row["endpoint_id"], row["sample_sha256"], row["query_position"], payload),
    )
    if cursor.rowcount == 1:
        return
    stored = connection.execute(
        "SELECT payload FROM query_outcomes WHERE query_key = ?", (key,)
    ).fetchone()
    if stored is None:
        raise RuntimeError("query-outcome index lost an existing key")
    reference = json.loads(stored[0])
    observed = json.loads(payload)
    for name in _QUERY_OUTCOME_FIELDS:
        left, right = reference[name], observed[name]
        equal = left == right
        if isinstance(left, float) and isinstance(right, float):
            equal = math.isclose(
                left, right, abs_tol=FLOAT32_ATOL, rel_tol=2e-5
            )
        if not equal:
            raise ValueError(
                f"query-level outcome differs across layer/head/parent: {name}"
            )


def _group_rows_from_query_index(
    connection: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], int]:
    group_rows: list[dict[str, Any]] = []
    current: tuple[str, str] | None = None
    response_rows: list[dict[str, Any]] = []
    answer_query_count = 0

    def flush() -> None:
        if not response_rows:
            return
        derived = _group_equal_response_summary(response_rows)
        if len(derived) != 1:
            raise ValueError("one endpoint/sample produced multiple response groups")
        group_rows.extend(derived)

    for endpoint, sample, payload in connection.execute(
        """SELECT endpoint_id, sample_sha256, payload FROM query_outcomes
           ORDER BY endpoint_id, sample_sha256, query_position"""
    ):
        identity = (str(endpoint), str(sample))
        if identity != current:
            flush()
            current = identity
            response_rows = []
        response_rows.append(json.loads(payload))
        answer_query_count += 1
    flush()
    return group_rows, answer_query_count


def write_artifacts(
    raw_rows: Iterable[Mapping[str, Any]], output_dir: Path
) -> dict[str, Any]:
    """Two-pass, disk-backed, representation-preserving formal reducer.

    Pass one validates and stages individual logical rows.  Pass two replays
    only the frozen ``endpoint_id/sample_sha256/row_ordinal`` order.  Therefore
    no cumulative sample-sized list or observation-derived logical-row ceiling
    exists, and floating reductions are independent of chunk completion order.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    spill = Path(tempfile.mkdtemp(prefix=".functional-stream.", dir=output_dir))
    connection = sqlite3.connect(spill / "formal_reducer.sqlite")
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """CREATE TABLE staged_rows (
               endpoint_row_id TEXT PRIMARY KEY,
               endpoint_id TEXT NOT NULL,
               task TEXT NOT NULL,
               sample_sha256 TEXT NOT NULL,
               content_group_sha256 TEXT NOT NULL,
               row_ordinal INTEGER NOT NULL CHECK(row_ordinal >= 0),
               logical_row_id TEXT NOT NULL,
               layer INTEGER NOT NULL,
               query_head INTEGER NOT NULL,
               query_position INTEGER NOT NULL,
               parent_position INTEGER NOT NULL,
               query_group TEXT NOT NULL,
               payload TEXT NOT NULL,
               UNIQUE(endpoint_id, sample_sha256, row_ordinal)
           );
           CREATE INDEX staged_formal_order ON staged_rows(
               endpoint_id, sample_sha256, row_ordinal
           );
           CREATE INDEX staged_query_order ON staged_rows(
               query_group, query_position, row_ordinal
           );
           CREATE TABLE query_statistics (
               query_group TEXT PRIMARY KEY,
               gamma_query_variance REAL NOT NULL,
               posterior_top1_changed INTEGER NOT NULL
           );
           CREATE TABLE query_outcomes (
               query_key TEXT PRIMARY KEY,
               endpoint_id TEXT NOT NULL,
               sample_sha256 TEXT NOT NULL,
               query_position INTEGER NOT NULL,
               payload TEXT NOT NULL
           );
           CREATE INDEX query_outcome_order ON query_outcomes(
               endpoint_id, sample_sha256, query_position
           );
           CREATE TABLE aggregate_group_flip (
               reducer TEXT NOT NULL,
               aggregate_key TEXT NOT NULL,
               content_group_sha256 TEXT NOT NULL,
               flip INTEGER NOT NULL,
               PRIMARY KEY(reducer, aggregate_key, content_group_sha256)
           );
           CREATE TABLE parent_geometry (
               parent_key BLOB PRIMARY KEY,
               seed INTEGER NOT NULL,
               checkpoint_arm TEXT NOT NULL,
               task TEXT NOT NULL,
               sample_sha256 TEXT NOT NULL,
               content_group_sha256 TEXT NOT NULL,
               layer INTEGER NOT NULL,
               parent_position INTEGER NOT NULL,
               payload TEXT NOT NULL,
               parent_record TEXT NOT NULL,
               reference_order TEXT NOT NULL
           );"""
    )
    fragment_paths: dict[tuple[Any, ...], Path] = {}
    fragment_handles: dict[tuple[Any, ...], Any] = {}
    try:
        (
            row_count,
            sample_ids,
            content_group_ids,
            seeds,
            cells,
            _partitions,
        ) = _stage_functional_rows(connection, raw_rows)
        _validate_staged_cartesian(connection)
        _prepare_query_statistics(connection)

        layer_reducer = _DiskBackedGroupedReducer(
            connection,
            "layer_head",
            (
                "seed", "checkpoint_arm", "inference_operator", "cell",
                "task", "lambda_value", "layer", "query_head", "kv_head",
            ),
        )
        topology_reducer = _DiskBackedGroupedReducer(
            connection,
            "topology",
            (
                "seed", "checkpoint_arm", "inference_operator", "cell",
                "task", "lambda_value", "topology",
            ),
        )
        lambda_reducer = _DiskBackedGroupedReducer(
            connection,
            "lambda",
            (
                "seed", "checkpoint_arm", "inference_operator", "cell",
                "task", "lambda_value",
            ),
        )
        contraction_combiner = _ContractionCombiner()

        for payload, variance, top1_changed in connection.execute(
            """SELECT staged.payload, statistics.gamma_query_variance,
                      statistics.posterior_top1_changed
               FROM staged_rows AS staged
               JOIN query_statistics AS statistics
                 ON statistics.query_group = staged.query_group
               ORDER BY staged.endpoint_id, staged.sample_sha256,
                        staged.row_ordinal"""
        ):
            row = json.loads(payload)
            row["gamma_query_variance"] = float(variance)
            row["posterior_top1_changed"] = bool(top1_changed)
            partition = tuple(row[name] for name in PARTITION_COLUMNS)
            path = fragment_paths.setdefault(
                partition,
                spill
                / (
                    hashlib.sha256(
                        canonical_json_bytes(list(partition))
                    ).hexdigest()
                    + ".jsonl"
                ),
            )
            handle = fragment_handles.get(partition)
            if handle is None:
                handle = path.open("ab")
                fragment_handles[partition] = handle
            handle.write(
                canonical_json_bytes(
                    {name: row[name] for name in OUTPUT_COLUMNS}
                )
            )
            layer_reducer.add(row)
            topology_reducer.add(row)
            lambda_reducer.add(row)
            _register_parent_geometry(
                connection,
                _parent_geometry_row(row),
                (str(row["inference_operator"]), float(row["lambda_value"])),
            )
            _register_query_outcome(connection, row)

        connection.commit()
        for (parent_record,) in connection.execute(
            """SELECT parent_record FROM parent_geometry
               ORDER BY seed, checkpoint_arm, task, sample_sha256,
                        content_group_sha256, layer, parent_position"""
        ):
            contraction_combiner.add(json.loads(parent_record))
        for handle in fragment_handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        fragment_handles.clear()

        parquet_path = output_dir / OUTPUT_NAMES["rows"]
        written = _write_bounded_parquet(
            parquet_path, _merge_partition_fragments(list(fragment_paths.values()))
        )
        if written != row_count:
            raise RuntimeError("functional detail merge changed the row count")
        aggregate_sets = {
            "layer_head": layer_reducer.finish(),
            "contraction": contraction_combiner.finish(),
            "topology": topology_reducer.finish(),
            "lambda": lambda_reducer.finish(),
        }
        hashes = {OUTPUT_NAMES["rows"]: sha256_file(parquet_path)}
        for name, records in aggregate_sets.items():
            path = output_dir / OUTPUT_NAMES[name]
            atomic_write(path, _csv_bytes(records))
            hashes[path.name] = sha256_file(path)
        group_rows, answer_query_count = _group_rows_from_query_index(connection)
        summary = _build_summary_from_group_rows(
            row_count=row_count,
            answer_query_count=answer_query_count,
            sample_ids=sample_ids,
            content_group_ids=content_group_ids,
            seeds=seeds,
            cells=cells,
            group_rows=group_rows,
            artifact_sha256=hashes,
        )
        atomic_write(
            output_dir / OUTPUT_NAMES["summary"], canonical_json_bytes(summary)
        )
        return summary
    finally:
        for handle in fragment_handles.values():
            handle.close()
        connection.close()
        shutil.rmtree(spill, ignore_errors=True)


RAW_BOUNDARY_REASONS = frozenset(
    {
        "zero_length_receiver_interval",
        "duplicate_or_overlap_receiver_offsets",
        "zero_length_source_interval",
        "exact_duplicate_source_offsets",
        "partial_overlap_source_offsets",
        "candidate_without_receiver_intersection",
    }
)
RAW_ORIGINS = frozenset(
    {"span_overlap", "window_neighbor", "fallback_or_unknown"}
)


def _span(value: Any, name: str, *, optional: bool = False) -> list[int] | None:
    if optional and value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a length-two span")
    return [
        _integer(value[0], f"{name}[0]"),
        _integer(value[1], f"{name}[1]"),
    ]


def validate_raw_topology_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one model-output-free raw-to-runtime topology parent row."""

    if set(raw) != set(RAW_TOPOLOGY_COLUMNS):
        missing = sorted(set(RAW_TOPOLOGY_COLUMNS) - set(raw))
        extra = sorted(set(raw) - set(RAW_TOPOLOGY_COLUMNS))
        raise ValueError(f"raw topology schema mismatch; missing={missing}, extra={extra}")
    row = {name: raw[name] for name in RAW_TOPOLOGY_COLUMNS}
    if _integer(row["schema_version"], "schema_version") != 1:
        raise ValueError("raw topology schema version mismatch")
    if row["split_role"] != SPLIT_ROLE:
        raise ValueError("raw topology data firewall accepts E0-design only")
    if row["task"] not in TASKS:
        raise ValueError("raw topology task is unknown")
    row["sample_sha256"] = _sha256(row["sample_sha256"], "sample_sha256")
    row["content_group_sha256"] = _sha256(
        row["content_group_sha256"], "content_group_sha256"
    )
    row["span_geometry_sha256"] = _sha256(
        row["span_geometry_sha256"], "span_geometry_sha256"
    )
    for name in (
        "parent_position",
        "receiver_token_id",
        "raw_candidate_count",
        "runtime_candidate_count",
        "candidate_window",
    ):
        row[name] = _integer(row[name], name)
    if not 2 <= row["raw_candidate_count"] <= 4:
        raise ValueError("raw topology requires 2<=raw_m<=4")
    if not 1 <= row["runtime_candidate_count"] <= 4:
        raise ValueError("raw topology requires 1<=runtime_m<=4")
    row["receiver_offset"] = _span(row["receiver_offset"], "receiver_offset")
    row["receiver_span"] = _span(row["receiver_span"], "receiver_span")
    assert row["receiver_span"] is not None
    if (
        row["receiver_span"][0] < 0
        or row["receiver_span"][1] < row["receiver_span"][0]
    ):
        raise ValueError("raw topology receiver span is reversed or negative")
    for name, count in (
        ("raw_candidate_indices", row["raw_candidate_count"]),
        ("runtime_candidate_indices", row["runtime_candidate_count"]),
    ):
        if not isinstance(row[name], (list, tuple)) or len(row[name]) != count:
            raise ValueError(f"{name} length differs from candidate count")
        row[name] = [_integer(value, f"{name}[]") for value in row[name]]
        if len(set(row[name])) != len(row[name]):
            raise ValueError(f"{name} contains duplicate indices")
    for name, count in (
        ("raw_weights", row["raw_candidate_count"]),
        ("runtime_weights", row["runtime_candidate_count"]),
    ):
        if not isinstance(row[name], (list, tuple)) or len(row[name]) != count:
            raise ValueError(f"{name} length differs from candidate count")
        row[name] = [
            _finite_float(value, f"{name}[]", nonnegative=True)
            for value in row[name]
        ]
        if any(value <= 0 for value in row[name]) or not math.isclose(
            sum(row[name]), 1.0, abs_tol=FLOAT32_ATOL, rel_tol=0
        ):
            raise ValueError(f"{name} must be a positive normalized distribution")
    if not isinstance(row["candidates"], list) or len(row["candidates"]) != row[
        "raw_candidate_count"
    ]:
        raise ValueError("raw topology candidate ledger length mismatch")
    candidates: list[dict[str, Any]] = []
    runtime_by_index = dict(
        zip(row["runtime_candidate_indices"], row["runtime_weights"])
    )
    for position, value in enumerate(row["candidates"]):
        if not isinstance(value, Mapping) or set(value) != set(RAW_CANDIDATE_COLUMNS):
            raise ValueError("raw topology candidate schema mismatch")
        candidate = {name: value[name] for name in RAW_CANDIDATE_COLUMNS}
        for name in ("slot", "source_index", "source_token_id", "intersection_length"):
            candidate[name] = _integer(candidate[name], f"candidate.{name}")
        if candidate["slot"] != position:
            raise ValueError("raw topology candidate slots are not canonical")
        if candidate["source_index"] != row["raw_candidate_indices"][position]:
            raise ValueError("raw topology candidate/index ledger differs")
        candidate["source_offset"] = _span(
            candidate["source_offset"], "candidate.source_offset"
        )
        candidate["source_span"] = _span(
            candidate["source_span"], "candidate.source_span"
        )
        candidate["intersection"] = _span(
            candidate["intersection"], "candidate.intersection", optional=True
        )
        if candidate["origin"] not in RAW_ORIGINS:
            raise ValueError("raw topology candidate origin is unknown")
        if candidate["origin"] == "window_neighbor" and row["candidate_window"] <= 0:
            raise ValueError("window-neighbor origin requires candidate_window>0")
        candidate["raw_weight"] = _finite_float(
            candidate["raw_weight"], "candidate.raw_weight", nonnegative=True
        )
        candidate["runtime_weight"] = _finite_float(
            candidate["runtime_weight"], "candidate.runtime_weight", nonnegative=True
        )
        for name in (
            "runtime_retained",
            "complete_receiver_explanation",
        ):
            if not isinstance(candidate[name], bool):
                raise ValueError(f"candidate.{name} must be boolean")
        expected_intersection = (
            max(row["receiver_span"][0], candidate["source_span"][0]),
            min(row["receiver_span"][1], candidate["source_span"][1]),
        )
        expected_intersection_value = (
            list(expected_intersection)
            if expected_intersection[1] > expected_intersection[0]
            else None
        )
        if candidate["intersection"] != expected_intersection_value:
            raise ValueError("raw topology stored intersection does not recompute")
        expected_length = (
            expected_intersection[1] - expected_intersection[0]
            if expected_intersection_value is not None
            else 0
        )
        if candidate["intersection_length"] != expected_length:
            raise ValueError("raw topology intersection length does not recompute")
        complete = expected_intersection_value == row["receiver_span"]
        if candidate["complete_receiver_explanation"] != complete:
            raise ValueError("raw topology complete-explanation flag does not recompute")
        retained = candidate["source_index"] in runtime_by_index
        if candidate["runtime_retained"] != retained or not math.isclose(
            candidate["runtime_weight"],
            runtime_by_index.get(candidate["source_index"], 0.0),
            abs_tol=FLOAT32_ATOL,
            rel_tol=0,
        ):
            raise ValueError("raw topology runtime retention/weight does not recompute")
        if not math.isclose(
            candidate["raw_weight"], row["raw_weights"][position],
            abs_tol=FLOAT32_ATOL, rel_tol=0,
        ):
            raise ValueError("raw topology candidate/raw weight ledger differs")
        candidates.append(candidate)
    row["candidates"] = candidates
    for name in (
        "certified",
        "offset_uncertified",
        "duplicate_or_overlap_alias",
        "runtime_functional_eligible",
        "functional_metrics_present",
    ):
        if not isinstance(row[name], bool):
            raise ValueError(f"raw topology {name} must be boolean")
    if row["offset_uncertified"] == row["certified"]:
        raise ValueError("raw topology certified/uncertified flags are inconsistent")
    if row["functional_metrics_present"]:
        raise ValueError("raw topology ledger must never contain functional metrics")
    expected_functional = row["certified"] and row["runtime_candidate_count"] >= 2
    if row["runtime_functional_eligible"] != expected_functional:
        raise ValueError("raw topology functional eligibility does not recompute")
    if not row["certified"] and row["runtime_candidate_count"] > 1:
        raise ValueError("uncertified raw parent retained multiple runtime candidates")
    if row["taxonomy"] not in TOPOLOGIES or row["taxonomy"] == "not_applicable":
        raise ValueError("raw topology taxonomy is invalid")
    source_spans = [tuple(candidate["source_span"]) for candidate in candidates]
    independent_complete = (
        all(candidate["complete_receiver_explanation"] for candidate in candidates)
        and len({candidate["source_token_id"] for candidate in candidates})
        == len(candidates)
        and len(set(source_spans)) == len(source_spans)
    )
    boundary = (
        row["certification_reason"] in RAW_BOUNDARY_REASONS
        or any(candidate["origin"] == "fallback_or_unknown" for candidate in candidates)
    )
    expected_taxonomy = classify_candidate_topology(
        tuple(row["receiver_span"]),
        source_spans,
        candidate_origins=[candidate["origin"] for candidate in candidates],
        boundary_or_fallback=boundary,
        independent_competitors=independent_complete,
        certified_partition=row["certified"],
        duplicate_or_overlap_alias=row["duplicate_or_overlap_alias"],
    )
    if row["taxonomy"] != expected_taxonomy:
        raise ValueError("raw topology taxonomy does not reproduce from frozen geometry")
    geometry_sha = hashlib.sha256(
        canonical_json_bytes(
            {"receiver_span": row["receiver_span"], "candidates": candidates}
        )
    ).hexdigest()
    if geometry_sha != row["span_geometry_sha256"]:
        raise ValueError("raw topology span geometry SHA does not recompute")
    return row


def read_raw_topology_sidecar(path: Path) -> list[dict[str, Any]]:
    """Read the A4 v2 model-output-free topology ledger, fail-closed.

    Historical v1 sidecars belong to abandoned executions and are deliberately
    rejected: A4 forbids both resume and artifact reuse.
    """

    import torch

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 2
        or payload.get("protocol_id")
        != "fpct_e1_e0_design_input_lock_v2_streaming"
        or payload.get("split_role") != SPLIT_ROLE
        or payload.get("status")
        != "GO_STREAMING_CPU_INPUT_LOCK_NO_MODEL_OUTPUT"
    ):
        raise ValueError("raw topology sidecar identity/firewall mismatch")
    firewall = payload.get("firewall")
    if not isinstance(firewall, Mapping) or any(
        firewall.get(name) is not False
        for name in (
            "e1_pilot_consumed",
            "model_selection_consumed",
            "test_consumed",
            "model_or_checkpoint_loaded",
            "gpu_or_cuda_used",
        )
    ):
        raise ValueError("raw topology sidecar identity/firewall mismatch")
    streaming = payload.get("streaming_contract")
    if (
        not isinstance(streaming, Mapping)
        or streaming.get("protocol_id")
        != "fpct_e1_mechanism_audit_v6_representation_preserving_streaming"
        or streaming.get("physical_chunk_rows") != 4096
        or streaming.get("expanded_logical_rows_present") is not False
        or not isinstance(streaming.get("geometry_lock"), Mapping)
        or not isinstance(streaming.get("streaming_template_lock"), Mapping)
    ):
        raise ValueError("raw topology sidecar lacks the operative A4 lock")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for item in payload.get("items", []):
        if not isinstance(item, Mapping):
            raise ValueError("raw topology sidecar item is not a mapping")
        for raw in item.get("raw_topology_ledger", []):
            row = validate_raw_topology_row(raw)
            if (
                row["task"] != item.get("task")
                or row["sample_sha256"] != item.get("sample_sha256")
                or row["content_group_sha256"]
                != item.get("content_group_sha256")
            ):
                raise ValueError("raw topology row differs from sidecar item identity")
            key = (
                row["task"],
                row["sample_sha256"],
                row["parent_position"],
            )
            if key in seen:
                raise ValueError("duplicate raw topology parent row")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("raw topology sidecar contains no raw m>=2 parents")
    return sorted(
        rows,
        key=lambda row: (
            row["task"],
            row["content_group_sha256"],
            row["sample_sha256"],
            row["parent_position"],
        ),
    )


def raw_topology_aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    keys = ("task", "taxonomy", "raw_candidate_count", "runtime_candidate_count")
    for row in rows:
        groups[tuple(row[name] for name in keys)].append(row)
    output = []
    for key, members in sorted(groups.items()):
        task, taxonomy, raw_m, runtime_m = key
        certified = sum(bool(row["certified"]) for row in members)
        output.append(
            {
                "schema_version": 1,
                "task": task,
                "taxonomy": taxonomy,
                "raw_m": raw_m,
                "runtime_m": runtime_m,
                "parent_count": len(members),
                "sample_count": len({row["sample_sha256"] for row in members}),
                "content_group_count": len(
                    {row["content_group_sha256"] for row in members}
                ),
                "raw_candidate_atom_count": sum(
                    int(row["raw_candidate_count"]) for row in members
                ),
                "runtime_candidate_atom_count": sum(
                    int(row["runtime_candidate_count"]) for row in members
                ),
                "raw_extra_slot_count": sum(
                    max(int(row["raw_candidate_count"]) - 1, 0)
                    for row in members
                ),
                "runtime_extra_slot_count": sum(
                    max(int(row["runtime_candidate_count"]) - 1, 0)
                    for row in members
                ),
                "raw_minus_runtime_candidate_count": sum(
                    int(row["raw_candidate_count"])
                    - int(row["runtime_candidate_count"])
                    for row in members
                ),
                "raw_minus_runtime_extra_slot_count": sum(
                    max(int(row["raw_candidate_count"]) - 1, 0)
                    - max(int(row["runtime_candidate_count"]) - 1, 0)
                    for row in members
                ),
                "certified_parent_count": certified,
                "offset_uncertified_parent_count": len(members) - certified,
                "functional_metric_contract": (
                    "SEPARATE_RUNTIME_FUNCTIONAL_TABLE_CERTIFIED_ONLY"
                    if certified == len(members)
                    else "UNAVAILABLE_UNCERTIFIED_NO_FUNCTIONAL_METRIC"
                ),
            }
        )
    return output


def raw_topology_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, Any] = {}
    for task in TASKS:
        members = [row for row in rows if row["task"] == task]
        if not members:
            continue
        taxonomy: dict[str, Any] = {}
        for name in sorted({str(row["taxonomy"]) for row in members}):
            selected = [row for row in members if row["taxonomy"] == name]
            taxonomy[name] = {
                "parent_count": len(selected),
                "sample_count": len({row["sample_sha256"] for row in selected}),
                "content_group_count": len(
                    {row["content_group_sha256"] for row in selected}
                ),
            }
        by_task[task] = {
            "parent_count": len(members),
            "sample_count": len({row["sample_sha256"] for row in members}),
            "content_group_count": len(
                {row["content_group_sha256"] for row in members}
            ),
            "raw_m_distribution": {
                str(value): sum(row["raw_candidate_count"] == value for row in members)
                for value in range(2, 5)
            },
            "runtime_m_distribution": {
                str(value): sum(
                    row["runtime_candidate_count"] == value for row in members
                )
                for value in range(1, 5)
            },
            "raw_candidate_atom_count": sum(
                int(row["raw_candidate_count"]) for row in members
            ),
            "runtime_candidate_atom_count": sum(
                int(row["runtime_candidate_count"]) for row in members
            ),
            "raw_minus_runtime_candidate_count": sum(
                int(row["raw_candidate_count"])
                - int(row["runtime_candidate_count"])
                for row in members
            ),
            "raw_extra_slot_count": sum(
                max(int(row["raw_candidate_count"]) - 1, 0)
                for row in members
            ),
            "runtime_extra_slot_count": sum(
                max(int(row["runtime_candidate_count"]) - 1, 0)
                for row in members
            ),
            "raw_minus_runtime_extra_slot_count": sum(
                max(int(row["raw_candidate_count"]) - 1, 0)
                - max(int(row["runtime_candidate_count"]) - 1, 0)
                for row in members
            ),
            "offset_uncertified_parent_count": sum(
                bool(row["offset_uncertified"]) for row in members
            ),
            "taxonomy": taxonomy,
        }
    return {
        "schema_version": 1,
        "protocol_id": "fpct_e1_raw_to_runtime_topology_audit_v1",
        "status": "COMPLETE_MODEL_OUTPUT_FREE",
        "split_role": SPLIT_ROLE,
        "row_count": len(rows),
        "task": by_task,
        "functional_metric_contract": (
            "raw ledger has no logits/KV/accuracy/mechanism values; uncertified "
            "parents have no functional metric; certified functional metrics live "
            "only in the separate runtime mechanism table"
        ),
        "e1_pilot_consumed": False,
        "confirmatory_consumed": False,
    }


def _atomic_raw_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the ledger one row at a time instead of materializing one large blob."""

    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for row in rows:
                handle.write(canonical_json_bytes(dict(row)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _raw_arrow_schema() -> Any:
    import pyarrow as pa

    span = pa.list_(pa.int64(), 2)
    candidate = pa.struct(
        [
            pa.field("slot", pa.int64(), nullable=False),
            pa.field("source_index", pa.int64(), nullable=False),
            pa.field("source_token_id", pa.int64(), nullable=False),
            pa.field("source_offset", span, nullable=False),
            pa.field("source_span", span, nullable=False),
            pa.field("intersection", span),
            pa.field("intersection_length", pa.int64(), nullable=False),
            pa.field("origin", pa.string(), nullable=False),
            pa.field("raw_weight", pa.float64(), nullable=False),
            pa.field("runtime_retained", pa.bool_(), nullable=False),
            pa.field("runtime_weight", pa.float64(), nullable=False),
            pa.field("complete_receiver_explanation", pa.bool_(), nullable=False),
        ]
    )
    types = {
        "schema_version": pa.int64(),
        "split_role": pa.string(),
        "task": pa.string(),
        "sample_sha256": pa.string(),
        "content_group_sha256": pa.string(),
        "parent_position": pa.int64(),
        "receiver_token_id": pa.int64(),
        "receiver_offset": span,
        "receiver_span": span,
        "raw_candidate_count": pa.int64(),
        "runtime_candidate_count": pa.int64(),
        "raw_candidate_indices": pa.list_(pa.int64()),
        "runtime_candidate_indices": pa.list_(pa.int64()),
        "raw_weights": pa.list_(pa.float64()),
        "runtime_weights": pa.list_(pa.float64()),
        "candidates": pa.list_(candidate),
        "certified": pa.bool_(),
        "offset_uncertified": pa.bool_(),
        "certification_reason": pa.string(),
        "taxonomy": pa.string(),
        "candidate_window": pa.int64(),
        "duplicate_or_overlap_alias": pa.bool_(),
        "runtime_functional_eligible": pa.bool_(),
        "functional_metrics_present": pa.bool_(),
        "span_geometry_sha256": pa.string(),
    }
    return pa.schema(
        [pa.field(name, types[name], nullable=False) for name in RAW_TOPOLOGY_COLUMNS]
    )


def _write_raw_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        schema = _raw_arrow_schema()
        with pq.ParquetWriter(
            temporary,
            schema,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
        ) as writer:
            for start in range(0, len(rows), 4096):
                table = pa.Table.from_pylist(
                    [
                        {name: row[name] for name in RAW_TOPOLOGY_COLUMNS}
                        for row in rows[start : start + 4096]
                    ],
                    schema=schema,
                )
                writer.write_table(table)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_raw_topology_artifacts(
    input_lock_sidecar: Path, output_dir: Path
) -> dict[str, Any]:
    rows = read_raw_topology_sidecar(input_lock_sidecar)
    aggregates = raw_topology_aggregate_rows(rows)
    summary = raw_topology_summary(rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        name: output_dir / filename
        for name, filename in RAW_TOPOLOGY_OUTPUT_NAMES.items()
    }
    _atomic_raw_jsonl(paths["jsonl"], rows)
    _write_raw_parquet(paths["parquet"], rows)
    atomic_write(paths["aggregate_csv"], _csv_bytes(aggregates))
    atomic_write(paths["aggregate_json"], canonical_json_bytes(summary))
    artifacts = {
        paths["jsonl"].name: {
            "sha256": sha256_file(paths["jsonl"]),
            "bytes": paths["jsonl"].stat().st_size,
            "row_count": len(rows),
        },
        paths["parquet"].name: {
            "sha256": sha256_file(paths["parquet"]),
            "bytes": paths["parquet"].stat().st_size,
            "row_count": len(rows),
        },
        paths["aggregate_csv"].name: {
            "sha256": sha256_file(paths["aggregate_csv"]),
            "bytes": paths["aggregate_csv"].stat().st_size,
            "row_count": len(aggregates),
        },
        paths["aggregate_json"].name: {
            "sha256": sha256_file(paths["aggregate_json"]),
            "bytes": paths["aggregate_json"].stat().st_size,
        },
    }
    manifest = {
        "schema_version": 1,
        "protocol_id": "fpct_e1_raw_to_runtime_topology_audit_v1",
        "status": "GO_MODEL_OUTPUT_FREE",
        "split_role": SPLIT_ROLE,
        "input_lock_sidecar": {
            "path": str(input_lock_sidecar),
            "sha256": sha256_file(input_lock_sidecar),
            "bytes": input_lock_sidecar.stat().st_size,
        },
        "artifacts": artifacts,
        "functional_metrics_present": False,
        "uncertified_functional_metrics_available": False,
        "runtime_functional_rows_are_separate": True,
        "e1_pilot_consumed": False,
        "confirmatory_consumed": False,
    }
    atomic_write(paths["manifest"], canonical_json_bytes(manifest))
    return manifest


def verify_raw_topology_artifacts(
    input_lock_sidecar: Path, output_dir: Path
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    rows = read_raw_topology_sidecar(input_lock_sidecar)
    aggregates = raw_topology_aggregate_rows(rows)
    summary = raw_topology_summary(rows)
    paths = {
        name: output_dir / filename
        for name, filename in RAW_TOPOLOGY_OUTPUT_NAMES.items()
    }
    with paths["jsonl"].open("rb") as handle:
        for row in rows:
            if handle.readline() != canonical_json_bytes(dict(row)):
                raise ValueError("raw topology JSONL does not reproduce")
        if handle.read(1):
            raise ValueError("raw topology JSONL contains trailing rows")
    parquet = pq.ParquetFile(paths["parquet"])
    if tuple(parquet.schema_arrow.names) != RAW_TOPOLOGY_COLUMNS:
        raise ValueError("raw topology parquet schema differs")
    expected = iter(rows)
    observed_count = 0
    for batch in parquet.iter_batches(batch_size=4096):
        columns = {
            name: batch.column(batch.schema.get_field_index(name))
            for name in RAW_TOPOLOGY_COLUMNS
        }
        for row_index in range(batch.num_rows):
            observed = {
                name: columns[name][row_index].as_py()
                for name in RAW_TOPOLOGY_COLUMNS
            }
            expected_row = next(expected, None)
            if expected_row is None or canonical_json_bytes(
                observed
            ) != canonical_json_bytes(expected_row):
                raise ValueError("raw topology parquet content differs")
            observed_count += 1
    if observed_count != len(rows):
        raise ValueError("raw topology parquet row count differs")
    if paths["aggregate_csv"].read_bytes() != _csv_bytes(aggregates):
        raise ValueError("raw topology CSV aggregate does not reproduce")
    with paths["aggregate_json"].open(encoding="utf-8") as handle:
        observed_aggregate = json.load(handle)
    if observed_aggregate != summary:
        raise ValueError("raw topology JSON aggregate does not reproduce")
    with paths["manifest"].open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        manifest.get("status") != "GO_MODEL_OUTPUT_FREE"
        or manifest.get("input_lock_sidecar", {}).get("sha256")
        != sha256_file(input_lock_sidecar)
    ):
        raise ValueError("raw topology manifest identity/provenance mismatch")
    for name in ("jsonl", "parquet", "aggregate_csv", "aggregate_json"):
        path = paths[name]
        record = manifest.get("artifacts", {}).get(path.name, {})
        if (
            record.get("sha256") != sha256_file(path)
            or record.get("bytes") != path.stat().st_size
        ):
            raise ValueError(f"raw topology artifact provenance mismatch: {path.name}")
    return {
        "status": "GO_MODEL_OUTPUT_FREE",
        "row_count": len(rows),
        "artifact_sha256": {
            path.name: sha256_file(path) for path in paths.values()
        },
        "e1_pilot_consumed": False,
        "confirmatory_consumed": False,
    }


def read_input_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one input row at a time; never materialize a stage artifact."""

    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        missing = [name for name in INPUT_COLUMNS if name not in parquet.schema_arrow.names]
        if missing:
            raise ValueError(f"mechanism parquet is missing input columns: {missing}")
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_ROWS):
            indices = {
                name: batch.schema.get_field_index(name) for name in INPUT_COLUMNS
            }
            columns = {name: batch.column(index) for name, index in indices.items()}
            for row_index in range(batch.num_rows):
                yield {
                    name: columns[name][row_index].as_py() for name in INPUT_COLUMNS
                }
        return
    else:
        raise ValueError("mechanism input must be .jsonl or .parquet")


def verify_artifacts(output_dir: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    paths = {name: output_dir / filename for name, filename in OUTPUT_NAMES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing E1 mechanism artifacts: {missing}")
    parquet = pq.ParquetFile(paths["rows"])
    if tuple(parquet.schema_arrow.names) != OUTPUT_COLUMNS:
        raise ValueError("mechanism parquet column order mismatch")
    temporary = Path(
        tempfile.mkdtemp(prefix=".mechanism-verify.", dir=output_dir.parent)
    )
    try:
        reproduced = write_artifacts(read_input_rows(paths["rows"]), temporary)
        for name, path in paths.items():
            candidate = temporary / path.name
            if (
                candidate.stat().st_size != path.stat().st_size
                or sha256_file(candidate) != sha256_file(path)
            ):
                raise ValueError(f"streaming artifact does not reproduce: {path.name}")
        with paths["summary"].open(encoding="utf-8") as handle:
            observed_summary = json.load(handle)
        if observed_summary != reproduced:
            raise ValueError("mechanism summary does not reproduce from frozen parquet")
        return {
            "status": "GO",
            "row_count": int(reproduced["row_count"]),
            "artifact_sha256": {
                path.name: sha256_file(path) for path in paths.values()
            },
            "e1_pilot_consumed": False,
            "confirmatory_consumed": False,
        }
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def synthetic_rows() -> list[dict[str, Any]]:
    """Small outcome-free fixture for CLI and schema verification."""

    sample = hashlib.sha256(b"fpct-e1-synthetic-sample").hexdigest()
    group = hashlib.sha256(b"fpct-e1-synthetic-group").hexdigest()
    input_sha = hashlib.sha256(b"fpct-e1-synthetic-input").hexdigest()
    alignment_sha = hashlib.sha256(b"fpct-e1-synthetic-alignment").hexdigest()
    labels_sha = hashlib.sha256(b"fpct-e1-synthetic-labels").hexdigest()
    response_sha = hashlib.sha256(b"The correct answer is A.").hexdigest()
    rows = []
    for checkpoint_arm in CHECKPOINT_ARMS:
        arm_offset = 0.02 if checkpoint_arm == "f_trained" else 0.0
        for operator in INFERENCE_OPERATORS:
            lambdas = (0.0,) if operator == "c_post" else LAMBDA_GRID
            for lambda_value in lambdas:
                for query_position in (4, 5):
                    prior = [0.6, 0.4]
                    if lambda_value == 0:
                        gamma = prior
                    elif query_position == 4:
                        gamma = [max(0.05, 0.6 - 0.25 * lambda_value), min(0.95, 0.4 + 0.25 * lambda_value)]
                    else:
                        gamma = [min(0.95, 0.6 + 0.25 * lambda_value), max(0.05, 0.4 - 0.25 * lambda_value)]
                    total = sum(gamma)
                    gamma = [value / total for value in gamma]
                    cpost_logp = -1.5 + arm_offset + 0.01 * query_position
                    delta = 0.0 if lambda_value == 0 else 0.005 * lambda_value * (1 if query_position == 5 else -1)
                    rows.append(attach_stream_identity({
                        "schema_version": SCHEMA_VERSION,
                        "split_role": "e0_design",
                        "seed": 2026072201,
                        "checkpoint_arm": checkpoint_arm,
                        "inference_operator": operator,
                        "cell": CELL_MAP[(checkpoint_arm, operator)],
                        "task": "ai2-arc",
                        "sample_sha256": sample,
                        "content_group_sha256": group,
                        "input_sha256": input_sha,
                        "alignment_sha256": alignment_sha,
                        "labels_sha256": labels_sha,
                        "gold_response_sha256": response_sha,
                        "layer": 0,
                        "query_head": 0,
                        "kv_head": 0,
                        "query_position": query_position,
                        "target_position": query_position + 1,
                        "target_token_id": 7 + query_position,
                        "parent_position": 2,
                        "candidate_count": 2,
                        "topology": "partition_compositional",
                        "lambda_value": lambda_value,
                        "prior": prior,
                        "candidate_indices": [17, 18, -1, -1],
                        "candidate_valid_mask": [True, True, False, False],
                        "candidate_slot_weights": [0.6, 0.4, 0.0, 0.0],
                        "statistical_weight": 1.0,
                        "gamma": gamma,
                        "source_d_k": 0.4,
                        "source_d_v": 0.3,
                        "source_energy_k": 2.0,
                        "source_energy_v": 1.5,
                        "fused_d_k": 0.1,
                        "fused_d_v": 0.15,
                        "fused_energy_k": 1.0,
                        "fused_energy_v": 1.0,
                        "candidate_logit_range": 0.2 * lambda_value,
                        "candidate_logit_variance": 0.01 * lambda_value * lambda_value,
                        "jensen_gap": 0.005 * lambda_value * lambda_value,
                        "parent_attention_mass": 0.25,
                        "output_delta_l2": 0.0 if lambda_value == 0 else 0.01 * lambda_value,
                        "gold_logp": cpost_logp + delta,
                        "cpost_gold_logp": cpost_logp,
                        "end_task_correct": True,
                        "cpost_end_task_correct": True,
                    }, query_position - 4))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry = subparsers.add_parser("dry-run", help="write synthetic artifacts only")
    dry.add_argument("--output-dir", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate", help="aggregate pre-captured E0-design rows")
    aggregate.add_argument("--input", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="independently reproduce frozen aggregates")
    verify.add_argument("--output-dir", type=Path, required=True)
    raw_topology = subparsers.add_parser(
        "raw-topology",
        help="export the model-output-free raw-to-runtime topology ledger",
    )
    raw_topology.add_argument("--input-lock-sidecar", type=Path, required=True)
    raw_topology.add_argument("--output-dir", type=Path, required=True)
    verify_raw = subparsers.add_parser(
        "verify-raw-topology",
        help="reproduce raw topology artifacts from the frozen input-lock sidecar",
    )
    verify_raw.add_argument("--input-lock-sidecar", type=Path, required=True)
    verify_raw.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "dry-run":
        result = write_artifacts(synthetic_rows(), args.output_dir)
    elif args.command == "aggregate":
        result = write_artifacts(read_input_rows(args.input), args.output_dir)
    elif args.command == "verify":
        result = verify_artifacts(args.output_dir)
    elif args.command == "raw-topology":
        result = write_raw_topology_artifacts(
            args.input_lock_sidecar, args.output_dir
        )
    else:
        result = verify_raw_topology_artifacts(
            args.input_lock_sidecar, args.output_dir
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
