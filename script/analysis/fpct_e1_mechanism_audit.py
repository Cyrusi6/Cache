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
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


SCHEMA_VERSION = 1
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

OUTPUT_NAMES = {
    "rows": "e1_mechanism_rows.parquet",
    "summary": "e1_mechanism_summary.json",
    "layer_head": "e1_layer_head_summary.csv",
    "contraction": "e1_projector_contraction.csv",
    "topology": "e1_candidate_topology.csv",
    "lambda": "e1_centered_lambda_summary.csv",
}

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
    "layer",
    "head",
    "query_position",
    "target_position",
    "parent_position",
    "candidate_count",
    "topology",
    "lambda_value",
    "prior",
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
    "prediction_correct",
    "cpost_prediction_correct",
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
    "accuracy_flip",
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
    active = native_valid[:, None, None, :] & torch.isfinite(mask)
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
    active = legal[:, None, None, :, :] & torch.isfinite(mask.unsqueeze(-1))
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
    parent_active = torch.isfinite(mask)
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
    if boundary_or_fallback or start < 0 or end <= start or any(a < 0 or b <= a for a, b in source_spans):
        return "boundary_fallback"
    if any(origin == "window_neighbor" for origin in origins):
        return "neighbor_expansion"
    intersections = sorted((max(start, a), min(end, b)) for a, b in source_spans if min(end, b) > max(start, a))
    if len(intersections) == len(source_spans):
        cursor = start
        partition = True
        for left, right in intersections:
            if left != cursor:
                partition = False
                break
            cursor = right
        if partition and cursor == end:
            return "partition_compositional"
    explicit_competition = independent_competitors or all(
        origin == "independent_overlap" for origin in origins
    )
    if explicit_competition and len(intersections) == len(source_spans):
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
    row["sample_sha256"] = _sha256(row["sample_sha256"], "sample_sha256")
    row["content_group_sha256"] = _sha256(row["content_group_sha256"], "content_group_sha256")
    for name in ("layer", "head", "query_position", "target_position", "parent_position", "candidate_count"):
        row[name] = _integer(row[name], name)
    if row["candidate_count"] > 4:
        raise ValueError("candidate_count exceeds frozen top-k=4")
    if row["target_position"] != row["query_position"] + 1:
        raise ValueError("teacher-forcing causal shift mismatch")
    if row["topology"] not in TOPOLOGIES:
        raise ValueError("unknown topology")
    if (row["candidate_count"] < 2) != (row["topology"] == "not_applicable"):
        raise ValueError("topology must be not_applicable exactly when m<2")
    row["lambda_value"] = _grid_value(row["lambda_value"])
    if row["inference_operator"] == "c_post" and row["lambda_value"] != 0.0:
        raise ValueError("C_post baseline is only represented at lambda=0")
    count = row["candidate_count"]
    row["prior"] = _validate_distribution(row["prior"], count, "prior")
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
    if not isinstance(row["prediction_correct"], bool) or not isinstance(row["cpost_prediction_correct"], bool):
        raise ValueError("correctness fields must be bool")

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
    row["accuracy_flip"] = row["prediction_correct"] != row["cpost_prediction_correct"]
    if row["lambda_value"] == 0.0:
        if abs(row["delta_gold_logp"]) > FLOAT32_ATOL or row["output_delta_l2"] > FLOAT32_ATOL:
            raise ValueError("lambda=0 violates exact C_post row oracle")
    return row


def _query_group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["seed"], row["checkpoint_arm"], row["inference_operator"], row["task"],
        row["sample_sha256"], row["layer"], row["head"], row["parent_position"],
        row["lambda_value"],
    )


def prepare_rows(raw_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [validate_and_derive_row(row) for row in raw_rows]
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
    rows.sort(key=lambda row: (
        row["seed"], row["checkpoint_arm"], row["inference_operator"], row["task"],
        row["content_group_sha256"], row["sample_sha256"], row["lambda_value"],
        row["layer"], row["head"], row["query_position"], row["parent_position"],
    ))
    return rows


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
        for name in MEAN_COLUMNS:
            record[f"mean_{name}"] = _mean(members, name)
        record["accuracy_flip_rate"] = _mean(members, "accuracy_flip")
        record["posterior_top1_change_rate"] = _mean(members, "posterior_top1_changed")
        output.append(record)
    return output


def layer_head_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return _group_rows(
        rows,
        ("seed", "checkpoint_arm", "inference_operator", "cell", "task", "lambda_value", "layer", "head"),
    )


def _consistent_parent_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate source/fused geometry repeated across query/operator/lambda."""

    keys = (
        "seed", "checkpoint_arm", "task", "sample_sha256", "content_group_sha256",
        "layer", "head", "parent_position",
    )
    metric_names = (
        "candidate_count", "topology", "source_d_k", "source_d_v", "source_energy_k",
        "source_energy_v", "fused_d_k", "fused_d_v", "fused_energy_k", "fused_energy_v",
        "normalized_source_d_k", "normalized_source_d_v", "normalized_fused_d_k",
        "normalized_fused_d_v", "projector_retention_k", "projector_retention_v",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, members in sorted(grouped.items()):
        reference = members[0]
        for other in members[1:]:
            for name in metric_names:
                if isinstance(reference[name], str):
                    equal = reference[name] == other[name]
                else:
                    equal = math.isclose(float(reference[name]), float(other[name]), abs_tol=FLOAT32_ATOL, rel_tol=2e-5)
                if not equal:
                    raise ValueError(f"pre-collapse candidate geometry differs across operator/query/lambda: {name}")
        output.append({**{name: value for name, value in zip(keys, key)}, **{name: reference[name] for name in metric_names}})
    return output


def contraction_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    parents = _consistent_parent_rows(rows)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ("seed", "checkpoint_arm", "task", "layer", "head", "topology")
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


def _write_parquet(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist([{name: row[name] for name in OUTPUT_COLUMNS} for row in rows])
    table = table.select(list(OUTPUT_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        pq.write_table(table, temporary, compression="zstd", use_dictionary=False, write_statistics=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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
            for name in ("gold_logp", "cpost_gold_logp", "prediction_correct", "cpost_prediction_correct"):
                if reference[name] != other[name] and not (
                    isinstance(reference[name], float)
                    and math.isclose(reference[name], other[name], abs_tol=FLOAT32_ATOL, rel_tol=2e-5)
                ):
                    raise ValueError(f"query-level outcome differs across layer/head/parent: {name}")
        unique.append({
            **{name: value for name, value in zip(keys, key)},
            "gold_logp": reference["gold_logp"],
            "cpost_gold_logp": reference["cpost_gold_logp"],
            "delta_gold_logp": reference["delta_gold_logp"],
            "prediction_correct": reference["prediction_correct"],
            "cpost_prediction_correct": reference["cpost_prediction_correct"],
            "accuracy_flip": reference["accuracy_flip"],
        })
    return unique


def build_summary(rows: Sequence[dict[str, Any]], artifact_sha256: Mapping[str, str]) -> dict[str, Any]:
    query_rows = _unique_query_summary(rows)
    primary_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    primary_keys = ("seed", "checkpoint_arm", "inference_operator", "cell", "task", "lambda_value")
    for row in query_rows:
        primary_groups[tuple(row[name] for name in primary_keys)].append(row)
    primary = []
    for key, members in sorted(primary_groups.items()):
        record = {name: value for name, value in zip(primary_keys, key)}
        record.update(
            answer_query_count=len(members),
            sample_count=len({row["sample_sha256"] for row in members}),
            content_group_count=len({row["content_group_sha256"] for row in members}),
            mean_gold_logp=_mean(members, "gold_logp"),
            mean_cpost_gold_logp=_mean(members, "cpost_gold_logp"),
            mean_delta_gold_logp=_mean(members, "delta_gold_logp"),
            accuracy=_mean(members, "prediction_correct"),
            cpost_accuracy=_mean(members, "cpost_prediction_correct"),
            accuracy_flip_rate=_mean(members, "accuracy_flip"),
        )
        primary.append(record)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "fpct_e1_e0_design_mechanism_audit_v1",
        "status": "COMPLETE",
        "split_role": SPLIT_ROLE,
        "lambda_grid": list(LAMBDA_GRID),
        "row_count": len(rows),
        "answer_query_count": len(query_rows),
        "sample_count": len({row["sample_sha256"] for row in rows}),
        "content_group_count": len({row["content_group_sha256"] for row in rows}),
        "seeds": sorted({row["seed"] for row in rows}),
        "cells": sorted({row["cell"] for row in rows}),
        "primary_teacher_forced_gold_logp": primary,
        "artifact_sha256": dict(sorted(artifact_sha256.items())),
        "integrity": {
            "only_e0_design_consumed": True,
            "e1_pilot_consumed": False,
            "confirmatory_consumed": False,
            "causal_shift_verified": True,
            "lambda_zero_exact_cpost_rows": True,
            "precollapse_geometry_operator_invariant": True,
        },
        "claim_boundary": "E0-design mechanism localization only; no E1-pilot or confirmatory performance claim.",
    }


def write_artifacts(raw_rows: Iterable[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    rows = prepare_rows(raw_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / OUTPUT_NAMES["rows"]
    _write_parquet(parquet_path, rows)
    aggregate_sets = {
        "layer_head": layer_head_records(rows),
        "contraction": contraction_records(rows),
        "topology": topology_records(rows),
        "lambda": lambda_records(rows),
    }
    hashes = {OUTPUT_NAMES["rows"]: sha256_file(parquet_path)}
    for name, records in aggregate_sets.items():
        path = output_dir / OUTPUT_NAMES[name]
        atomic_write(path, _csv_bytes(records))
        hashes[path.name] = sha256_file(path)
    summary = build_summary(rows, hashes)
    atomic_write(output_dir / OUTPUT_NAMES["summary"], canonical_json_bytes(summary))
    return summary


def read_input_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    raise ValueError("mechanism input must be .jsonl or .parquet")


def verify_artifacts(output_dir: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    paths = {name: output_dir / filename for name, filename in OUTPUT_NAMES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing E1 mechanism artifacts: {missing}")
    table = pq.read_table(paths["rows"])
    if tuple(table.column_names) != OUTPUT_COLUMNS:
        raise ValueError("mechanism parquet column order mismatch")
    stored_rows = table.to_pylist()
    prepared = prepare_rows([{name: row[name] for name in INPUT_COLUMNS} for row in stored_rows])
    for expected, observed in zip(prepared, stored_rows):
        for name in DERIVED_COLUMNS:
            if isinstance(expected[name], float):
                if not math.isclose(expected[name], observed[name], abs_tol=1e-12, rel_tol=1e-9):
                    raise ValueError(f"derived parquet field mismatch: {name}")
            elif expected[name] != observed[name]:
                raise ValueError(f"derived parquet field mismatch: {name}")
    expected_csv = {
        "layer_head": layer_head_records(prepared),
        "contraction": contraction_records(prepared),
        "topology": topology_records(prepared),
        "lambda": lambda_records(prepared),
    }
    for name, records in expected_csv.items():
        if paths[name].read_bytes() != _csv_bytes(records):
            raise ValueError(f"aggregate CSV mismatch: {paths[name].name}")
    hashes = {paths["rows"].name: sha256_file(paths["rows"])}
    for name in expected_csv:
        hashes[paths[name].name] = sha256_file(paths[name])
    expected_summary = build_summary(prepared, hashes)
    observed_summary = json.loads(paths["summary"].read_text())
    if observed_summary != expected_summary:
        raise ValueError("mechanism summary does not reproduce from frozen parquet")
    return {
        "status": "GO",
        "row_count": len(prepared),
        "artifact_sha256": {path.name: sha256_file(path) for path in paths.values()},
        "e1_pilot_consumed": False,
        "confirmatory_consumed": False,
    }


def synthetic_rows() -> list[dict[str, Any]]:
    """Small outcome-free fixture for CLI and schema verification."""

    sample = hashlib.sha256(b"fpct-e1-synthetic-sample").hexdigest()
    group = hashlib.sha256(b"fpct-e1-synthetic-group").hexdigest()
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
                    rows.append({
                        "schema_version": 1,
                        "split_role": "e0_design",
                        "seed": 2026072201,
                        "checkpoint_arm": checkpoint_arm,
                        "inference_operator": operator,
                        "cell": CELL_MAP[(checkpoint_arm, operator)],
                        "task": "ai2-arc",
                        "sample_sha256": sample,
                        "content_group_sha256": group,
                        "layer": 0,
                        "head": 0,
                        "query_position": query_position,
                        "target_position": query_position + 1,
                        "parent_position": 2,
                        "candidate_count": 2,
                        "topology": "partition_compositional",
                        "lambda_value": lambda_value,
                        "prior": prior,
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
                        "prediction_correct": True,
                        "cpost_prediction_correct": True,
                    })
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
    args = parser.parse_args(argv)
    if args.command == "dry-run":
        result = write_artifacts(synthetic_rows(), args.output_dir)
    elif args.command == "aggregate":
        result = write_artifacts(read_input_rows(args.input), args.output_dir)
    else:
        result = verify_artifacts(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
