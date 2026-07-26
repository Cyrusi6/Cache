from __future__ import annotations

"""Online, explicit-lifecycle instrumentation for FPCT query-time behavior."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


FPCT_CAPTURE_MODES = frozenset(
    {"prefill", "teacher_forced", "teacher_forced_response", "greedy_decode"}
)


def teacher_forced_query_mask(labels: Tensor) -> Tensor:
    """Return query eligibility where query ``t`` predicts non-ignored label ``t+1``."""

    if labels.ndim != 2:
        raise ValueError("teacher-forced labels must be [B,T]")
    eligible = torch.zeros_like(labels, dtype=torch.bool)
    if labels.shape[1] > 1:
        eligible[:, :-1] = labels[:, 1:] != -100
    return eligible


@dataclass
class _Moments:
    """Welford moments whose observations are reduced over query dimension 2."""

    count: Tensor
    mean: Tensor
    m2: Tensor
    minimum: Tensor
    maximum: Tensor

    @classmethod
    def from_values(cls, values: Tensor, valid: Tensor) -> "_Moments":
        values = values.detach().float()
        valid = torch.broadcast_to(valid.detach().bool(), values.shape)
        count = valid.sum(dim=2).float()
        safe = torch.where(valid, values, torch.zeros_like(values))
        mean = safe.sum(dim=2) / count.clamp_min(1)
        centered = torch.where(
            valid, values - mean.unsqueeze(2), torch.zeros_like(values)
        )
        m2 = centered.square().sum(dim=2)
        minimum = torch.where(
            valid, values, torch.full_like(values, torch.inf)
        ).amin(dim=2)
        maximum = torch.where(
            valid, values, torch.full_like(values, -torch.inf)
        ).amax(dim=2)
        minimum = torch.where(count > 0, minimum, torch.full_like(minimum, torch.inf))
        maximum = torch.where(count > 0, maximum, torch.full_like(maximum, -torch.inf))
        return cls(count, mean, m2, minimum, maximum)

    def pad_source(self, source_length: int) -> None:
        current = self.count.shape[2]
        if source_length < current:
            raise ValueError("FPCT capture source length cannot shrink")
        if source_length == current:
            return

        def extend(value: Tensor, fill: float) -> Tensor:
            shape = list(value.shape)
            shape[2] = source_length - current
            tail = torch.full(
                shape, fill, device=value.device, dtype=value.dtype
            )
            return torch.cat((value, tail), dim=2)

        self.count = extend(self.count, 0.0)
        self.mean = extend(self.mean, 0.0)
        self.m2 = extend(self.m2, 0.0)
        self.minimum = extend(self.minimum, torch.inf)
        self.maximum = extend(self.maximum, -torch.inf)

    def merge(self, other: "_Moments") -> None:
        if self.count.shape != other.count.shape:
            raise ValueError("FPCT capture moment shape changed")
        total = self.count + other.count
        delta = other.mean - self.mean
        cross = delta.square() * self.count * other.count / total.clamp_min(1)
        self.mean = torch.where(
            total > 0,
            self.mean + delta * other.count / total.clamp_min(1),
            self.mean,
        )
        self.m2 = self.m2 + other.m2 + cross
        self.minimum = torch.minimum(self.minimum, other.minimum)
        self.maximum = torch.maximum(self.maximum, other.maximum)
        self.count = total


@dataclass
class _ScalarMoments:
    count: int = 0
    mean: Tensor | None = None
    m2: Tensor | None = None
    minimum: Tensor | None = None
    maximum: Tensor | None = None

    def update(self, value: Tensor) -> None:
        value = value.detach().float()
        if value.numel() != 1:
            raise ValueError("FPCT scalar metric must contain one value")
        if self.mean is None:
            self.count = 1
            self.mean = value.clone()
            self.m2 = torch.zeros_like(value)
            self.minimum = value.clone()
            self.maximum = value.clone()
            return
        self.count += 1
        delta = value - self.mean
        self.mean = self.mean + delta / float(self.count)
        self.m2 = self.m2 + delta * (value - self.mean)
        self.minimum = torch.minimum(self.minimum, value)
        self.maximum = torch.maximum(self.maximum, value)


class _LayerCapture:
    def __init__(self, layer_index: int) -> None:
        self.layer_index = int(layer_index)
        self.forward_count = 0
        self.query_count = 0
        self.top_k: int | None = None
        self.batch_size: int | None = None
        self.num_heads: int | None = None
        self.parent_metrics: dict[str, _Moments] = {}
        self.gamma: _Moments | None = None
        self.scalar_metrics: dict[str, _ScalarMoments] = {}
        self.first_top1: Tensor | None = None
        self.top1_comparisons: Tensor | None = None
        self.top1_disagreements: Tensor | None = None
        self.top1_any_change: Tensor | None = None
        self.candidate_count: Tensor | None = None
        self.duplicate_atom_max: Tensor | None = None
        self.query_chunks: list[dict[str, Any]] = []

    @staticmethod
    def _extend_source(value: Tensor, source_length: int, fill: int | bool) -> Tensor:
        current = value.shape[-1]
        if source_length < current:
            raise ValueError("FPCT capture source length cannot shrink")
        if source_length == current:
            return value
        shape = list(value.shape)
        shape[-1] = source_length - current
        return torch.cat(
            (
                value,
                torch.full(shape, fill, device=value.device, dtype=value.dtype),
            ),
            dim=-1,
        )

    def _ensure_structure(
        self, *, batch_size: int, num_heads: int, source_length: int, top_k: int, device
    ) -> None:
        if self.batch_size is None:
            self.batch_size = batch_size
            self.num_heads = num_heads
            self.top_k = top_k
            shape = (batch_size, num_heads, source_length)
            self.first_top1 = torch.full(shape, -1, device=device, dtype=torch.long)
            self.top1_comparisons = torch.zeros(shape, device=device, dtype=torch.long)
            self.top1_disagreements = torch.zeros(shape, device=device, dtype=torch.long)
            self.top1_any_change = torch.zeros(shape, device=device, dtype=torch.bool)
            self.candidate_count = torch.zeros(
                batch_size, source_length, device=device, dtype=torch.long
            )
            return
        if (
            self.batch_size != batch_size
            or self.num_heads != num_heads
            or self.top_k != top_k
        ):
            raise ValueError("FPCT capture batch/head/top-k structure changed")
        assert self.first_top1 is not None
        assert self.top1_comparisons is not None
        assert self.top1_disagreements is not None
        assert self.top1_any_change is not None
        assert self.candidate_count is not None
        self.first_top1 = self._extend_source(self.first_top1, source_length, -1)
        self.top1_comparisons = self._extend_source(
            self.top1_comparisons, source_length, 0
        )
        self.top1_disagreements = self._extend_source(
            self.top1_disagreements, source_length, 0
        )
        self.top1_any_change = self._extend_source(
            self.top1_any_change, source_length, False
        )
        self.candidate_count = self._extend_source(
            self.candidate_count, source_length, 0
        )
        if self.gamma is not None:
            self.gamma.pad_source(source_length)
        for state in self.parent_metrics.values():
            state.pad_source(source_length)

    @staticmethod
    def _dense_candidates(payload: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        gamma = payload["gamma"].detach().float()
        candidate_mask = payload["candidate_mask"].detach().bool()
        parent = payload["parent_index"].detach().long().clamp_min(0)
        candidate = payload["candidate_index"].detach().long().clamp_min(0)
        top_k = int(payload["top_k"])
        source_length = int(payload["source_length"])
        b, h, q, memory = gamma.shape
        flat_index = (parent * top_k + candidate)[:, None, None, :].expand(
            b, h, q, memory
        )
        dense_size = source_length * top_k
        dense_gamma = torch.zeros(
            b, h, q, dense_size, device=gamma.device, dtype=torch.float32
        )
        dense_gamma.scatter_add_(
            3, flat_index, torch.where(candidate_mask, gamma, torch.zeros_like(gamma))
        )
        dense_valid_count = torch.zeros(
            b, h, q, dense_size, device=gamma.device, dtype=torch.long
        )
        dense_valid_count.scatter_add_(3, flat_index, candidate_mask.long())
        dense_valid = dense_valid_count > 0
        return (
            dense_gamma.reshape(b, h, q, source_length, top_k),
            dense_valid.reshape(b, h, q, source_length, top_k),
            dense_valid_count.reshape(b, h, q, source_length, top_k),
        )

    def update(
        self,
        scalar_metrics: Mapping[str, Tensor],
        payload: Mapping[str, Any],
        query_eligible: Tensor,
    ) -> None:
        gamma, gamma_valid, duplicate_count = self._dense_candidates(payload)
        duplicate_max = duplicate_count.amax().detach()
        self.duplicate_atom_max = (
            duplicate_max
            if self.duplicate_atom_max is None
            else torch.maximum(self.duplicate_atom_max, duplicate_max)
        )
        b, h, q, source_length, top_k = gamma.shape
        if query_eligible.shape != (b, q):
            raise ValueError("FPCT capture query mask slice must be [B,Q]")
        query_eligible = query_eligible.to(device=gamma.device, dtype=torch.bool)
        gamma_valid = gamma_valid & query_eligible[:, None, :, None, None]
        self._ensure_structure(
            batch_size=b,
            num_heads=h,
            source_length=source_length,
            top_k=top_k,
            device=gamma.device,
        )
        self.forward_count += 1
        start_query = self.query_count
        self.query_count += q

        gamma_batch = _Moments.from_values(gamma, gamma_valid)
        if self.gamma is None:
            self.gamma = gamma_batch
        else:
            self.gamma.merge(gamma_batch)

        parent_valid = gamma_valid.sum(dim=-1) >= 2
        assert self.candidate_count is not None
        observed_count = gamma_valid.sum(dim=-1).amax(dim=(1, 2))
        self.candidate_count = torch.maximum(self.candidate_count, observed_count)

        parent_metric_values = payload["parent_metrics"]
        for name, value in parent_metric_values.items():
            batch = _Moments.from_values(value, parent_valid)
            if name not in self.parent_metrics:
                self.parent_metrics[name] = batch
            else:
                self.parent_metrics[name].merge(batch)

        parent_metric_names = set(parent_metric_values)
        for name, value in scalar_metrics.items():
            if name in parent_metric_names or name == "gamma_query_variance":
                continue
            self.scalar_metrics.setdefault(name, _ScalarMoments()).update(value)

        negative = torch.full_like(gamma, -torch.inf)
        top1 = torch.where(gamma_valid, gamma, negative).argmax(dim=-1)
        valid_parent = parent_valid
        assert self.first_top1 is not None
        assert self.top1_comparisons is not None
        assert self.top1_disagreements is not None
        assert self.top1_any_change is not None
        prior_observed = self.first_top1 >= 0
        any_current = valid_parent.any(dim=2)
        first_index = valid_parent.long().argmax(dim=2)
        current_first = torch.gather(
            top1, 2, first_index.unsqueeze(2)
        ).squeeze(2)
        reference = torch.where(prior_observed, self.first_top1, current_first)
        query_axis = torch.arange(q, device=gamma.device)[None, None, :, None]
        comparable = valid_parent & (
            prior_observed.unsqueeze(2) | (query_axis > first_index.unsqueeze(2))
        )
        disagreement = comparable & (top1 != reference.unsqueeze(2))
        self.top1_comparisons += comparable.sum(dim=2)
        self.top1_disagreements += disagreement.sum(dim=2)
        self.top1_any_change |= disagreement.any(dim=2)
        self.first_top1 = torch.where(
            (~prior_observed) & any_current, current_first, self.first_top1
        )

        query_chunk: dict[str, Any] = {
            "positions": torch.arange(
                start_query, start_query + q, device=gamma.device, dtype=torch.long
            ),
            "eligible": query_eligible.detach(),
            "parent_count": parent_valid.sum(dim=-1).detach(),
            "top1_comparisons": comparable.sum(dim=-1).detach(),
            "top1_disagreements": disagreement.sum(dim=-1).detach(),
            "metrics": {},
        }
        for name, value in parent_metric_values.items():
            count = parent_valid.sum(dim=-1)
            mean = torch.where(parent_valid, value, torch.zeros_like(value)).sum(
                dim=-1
            ) / count.clamp_min(1)
            query_chunk["metrics"][name] = mean.detach()
        self.query_chunks.append(query_chunk)


def _number(value: Tensor) -> float:
    return float(value.detach().cpu())


def _moment_statistics(state: _Moments, *, head: int | None = None) -> dict[str, Any]:
    count = state.count if head is None else state.count[:, head]
    mean = state.mean if head is None else state.mean[:, head]
    m2 = state.m2 if head is None else state.m2[:, head]
    minimum = state.minimum if head is None else state.minimum[:, head]
    maximum = state.maximum if head is None else state.maximum[:, head]
    total = count.sum()
    aggregate_mean = (count * mean).sum() / total.clamp_min(1)
    aggregate_m2 = (
        m2 + count * (mean - aggregate_mean).square()
    ).sum()
    valid = count > 0
    aggregate_minimum = torch.where(
        valid, minimum, torch.full_like(minimum, torch.inf)
    ).amin()
    aggregate_maximum = torch.where(
        valid, maximum, torch.full_like(maximum, -torch.inf)
    ).amax()
    has = bool(valid.any())
    return {
        "count": int(_number(total)),
        "mean": _number(aggregate_mean) if has else 0.0,
        "variance": _number(aggregate_m2 / total.clamp_min(1)) if has else 0.0,
        "min": _number(aggregate_minimum) if has else 0.0,
        "max": _number(aggregate_maximum) if has else 0.0,
    }


def _scalar_statistics(state: _ScalarMoments) -> dict[str, Any]:
    assert state.mean is not None
    assert state.m2 is not None
    assert state.minimum is not None
    assert state.maximum is not None
    return {
        "count": state.count,
        "mean": _number(state.mean),
        "variance": _number(state.m2 / float(max(state.count, 1))),
        "min": _number(state.minimum),
        "max": _number(state.maximum),
    }


def _gamma_variance(state: _Moments, *, head: int | None = None) -> dict[str, Any]:
    count = state.count if head is None else state.count[:, head]
    m2 = state.m2 if head is None else state.m2[:, head]
    valid = count >= 2
    variance = m2 / count.clamp_min(1)
    stream_count = valid.sum()
    mean = torch.where(valid, variance, torch.zeros_like(variance)).sum() / stream_count.clamp_min(1)
    maximum = torch.where(
        valid, variance, torch.full_like(variance, -torch.inf)
    ).amax()
    has = bool(valid.any())
    return {
        "stream_count": int(_number(stream_count)),
        "mean": _number(mean) if has else 0.0,
        "max": _number(maximum) if has else 0.0,
    }


class FPCTCaptureAccumulator:
    """Capture query-time summaries across forwards without retaining K/V tensors."""

    def __init__(
        self,
        mode: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        query_mask: Tensor | None = None,
    ) -> None:
        normalized = str(mode).lower()
        if normalized not in FPCT_CAPTURE_MODES:
            raise ValueError(
                f"FPCT capture mode must be one of {sorted(FPCT_CAPTURE_MODES)}"
            )
        self.mode = normalized
        self.metadata = deepcopy(dict(metadata or {}))
        if query_mask is not None:
            if query_mask.ndim != 2:
                raise ValueError("FPCT capture query_mask must be [B,T]")
            self.query_mask = query_mask.detach().bool().clone()
        else:
            self.query_mask = None
        if self.mode in {"teacher_forced", "teacher_forced_response"} and self.query_mask is None:
            raise ValueError(
                "teacher-forced FPCT capture requires an explicit causal-shift query_mask"
            )
        self.layers: dict[int, _LayerCapture] = {}
        self.closed = False

    def update(
        self,
        layer_index: int,
        scalar_metrics: Mapping[str, Tensor],
        payload: Mapping[str, Any],
    ) -> None:
        if self.closed:
            raise RuntimeError("cannot update a closed FPCT capture")
        layer = self.layers.setdefault(int(layer_index), _LayerCapture(layer_index))
        query_length = int(payload["gamma"].shape[2])
        batch_size = int(payload["gamma"].shape[0])
        if self.query_mask is None:
            eligible = torch.ones(
                batch_size,
                query_length,
                device=payload["gamma"].device,
                dtype=torch.bool,
            )
        else:
            start = layer.query_count
            end = start + query_length
            if self.query_mask.shape[0] != batch_size or end > self.query_mask.shape[1]:
                raise ValueError("FPCT capture query_mask does not cover this forward")
            eligible = self.query_mask[:, start:end].to(payload["gamma"].device)
        layer.update(scalar_metrics, payload, eligible)

    def finalize(self) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("FPCT capture has already ended")
        self.closed = True
        layer_reports: dict[str, Any] = {}
        global_metric_parts: dict[str, list[dict[str, Any]]] = {}
        global_gamma_weighted = 0.0
        global_gamma_streams = 0
        global_gamma_max = 0.0
        top1_comparisons = 0
        top1_disagreements = 0
        any_top1_change = False

        for layer_index in sorted(self.layers):
            layer = self.layers[layer_index]
            if self.query_mask is not None and layer.query_count != self.query_mask.shape[1]:
                raise ValueError(
                    "FPCT capture ended before consuming the full query_mask"
                )
            assert layer.duplicate_atom_max is not None
            if _number(layer.duplicate_atom_max) > 1:
                raise ValueError(
                    "duplicate FPCT parent/candidate atom in capture payload"
                )
            metric_statistics = {
                name: _moment_statistics(state)
                for name, state in sorted(layer.parent_metrics.items())
            }
            metric_statistics.update(
                {
                    name: _scalar_statistics(state)
                    for name, state in sorted(layer.scalar_metrics.items())
                }
            )
            for name, stats in metric_statistics.items():
                global_metric_parts.setdefault(name, []).append(stats)
            assert layer.gamma is not None
            gamma = _gamma_variance(layer.gamma)
            global_gamma_weighted += gamma["mean"] * gamma["stream_count"]
            global_gamma_streams += gamma["stream_count"]
            global_gamma_max = max(global_gamma_max, gamma["max"])
            assert layer.top1_comparisons is not None
            assert layer.top1_disagreements is not None
            assert layer.top1_any_change is not None
            comparisons = int(_number(layer.top1_comparisons.sum()))
            disagreements = int(_number(layer.top1_disagreements.sum()))
            top1_comparisons += comparisons
            top1_disagreements += disagreements
            layer_any_change = bool(layer.top1_any_change.any())
            any_top1_change = any_top1_change or layer_any_change

            head_reports = []
            assert layer.num_heads is not None
            for head in range(layer.num_heads):
                head_gamma = _gamma_variance(layer.gamma, head=head)
                head_metrics = {
                    name: _moment_statistics(state, head=head)
                    for name, state in sorted(layer.parent_metrics.items())
                }
                head_reports.append(
                    {
                        "head_index": head,
                        "gamma_query_variance": head_gamma["mean"],
                        "gamma_query_variance_max": head_gamma["max"],
                        "metrics": {
                            name: stats["mean"] for name, stats in head_metrics.items()
                        },
                        "metric_statistics": head_metrics,
                    }
                )

            query_rows: list[dict[str, Any]] = []
            for chunk in layer.query_chunks:
                positions = chunk["positions"].detach().cpu().tolist()
                eligible = chunk["eligible"].detach().cpu()
                parent_count = chunk["parent_count"].detach().cpu()
                comparisons_tensor = chunk["top1_comparisons"].detach().cpu()
                disagreements_tensor = chunk["top1_disagreements"].detach().cpu()
                metric_tensors = {
                    name: value.detach().cpu()
                    for name, value in chunk["metrics"].items()
                }
                batch_size, num_heads, query_length = parent_count.shape
                for batch in range(batch_size):
                    for head in range(num_heads):
                        for local_query in range(query_length):
                            if not bool(eligible[batch, local_query]):
                                continue
                            query_rows.append(
                                {
                                    "batch_index": batch,
                                    "head_index": head,
                                    "query_position": positions[local_query],
                                    "parent_count": int(parent_count[batch, head, local_query]),
                                    "top1_comparisons": int(comparisons_tensor[batch, head, local_query]),
                                    "top1_disagreements": int(disagreements_tensor[batch, head, local_query]),
                                    "metrics": {
                                        name: float(value[batch, head, local_query])
                                        for name, value in metric_tensors.items()
                                    },
                                }
                            )

            assert layer.candidate_count is not None
            candidate_count_cpu = layer.candidate_count.detach().cpu()
            gamma_count_cpu = layer.gamma.count.detach().cpu()
            gamma_mean_cpu = layer.gamma.mean.detach().cpu()
            gamma_m2_cpu = layer.gamma.m2.detach().cpu()
            comparison_cpu = layer.top1_comparisons.detach().cpu()
            disagreement_cpu = layer.top1_disagreements.detach().cpu()
            any_change_cpu = layer.top1_any_change.detach().cpu()
            parent_metric_cpu = {
                name: (
                    state.count.detach().cpu(),
                    state.mean.detach().cpu(),
                    state.m2.detach().cpu(),
                )
                for name, state in layer.parent_metrics.items()
            }
            parent_rows: list[dict[str, Any]] = []
            assert layer.batch_size is not None
            for batch in range(layer.batch_size):
                for head in range(layer.num_heads):
                    for parent in range(candidate_count_cpu.shape[1]):
                        candidate_count = int(candidate_count_cpu[batch, parent])
                        if candidate_count < 2:
                            continue
                        stream_count = gamma_count_cpu[batch, head, parent]
                        stream_valid = stream_count >= 2
                        stream_variance = gamma_m2_cpu[batch, head, parent] / stream_count.clamp_min(1)
                        query_count = int(stream_count.max())
                        parent_rows.append(
                            {
                                "batch_index": batch,
                                "head_index": head,
                                "parent_position": parent,
                                "candidate_count": candidate_count,
                                "query_count": query_count,
                                "candidate_gamma_moments": [
                                    {
                                        "candidate_index": candidate,
                                        "count": int(stream_count[candidate]),
                                        "mean": float(
                                            gamma_mean_cpu[
                                                batch, head, parent, candidate
                                            ]
                                        ),
                                        "variance": float(
                                            gamma_m2_cpu[batch, head, parent, candidate]
                                            / stream_count[candidate].clamp_min(1)
                                        ),
                                    }
                                    for candidate in range(stream_count.shape[0])
                                    if int(stream_count[candidate]) > 0
                                ],
                                "gamma_query_variance": float(
                                    torch.where(
                                        stream_valid,
                                        stream_variance,
                                        torch.zeros_like(stream_variance),
                                    ).sum()
                                    / stream_valid.sum().clamp_min(1)
                                ),
                                "posterior_top1_change_rate": (
                                    int(disagreement_cpu[batch, head, parent])
                                    / int(comparison_cpu[batch, head, parent])
                                    if int(comparison_cpu[batch, head, parent])
                                    else 0.0
                                ),
                                "posterior_top1_any_change": bool(
                                    any_change_cpu[batch, head, parent]
                                ),
                                "metrics": {
                                    name: float(mean[batch, head, parent])
                                    for name, (count, mean, _m2) in parent_metric_cpu.items()
                                    if count[batch, head, parent] > 0
                                },
                                "metric_variances": {
                                    name: float(
                                        m2[batch, head, parent]
                                        / count[batch, head, parent].clamp_min(1)
                                    )
                                    for name, (count, _mean, m2) in parent_metric_cpu.items()
                                    if count[batch, head, parent] > 0
                                },
                            }
                        )

            layer_report = {
                "forward_count": layer.forward_count,
                "query_count": layer.query_count,
                "eligible_query_count": len(query_rows) // layer.num_heads,
                "gamma_query_variance": gamma["mean"],
                "gamma_query_variance_max": gamma["max"],
                "gamma_query_stream_count": gamma["stream_count"],
                "posterior_top1_change_rate": (
                    disagreements / comparisons if comparisons else 0.0
                ),
                "posterior_top1_any_change": layer_any_change,
                "metrics": {
                    name: stats["mean"] for name, stats in metric_statistics.items()
                },
                "metric_statistics": metric_statistics,
                "heads": head_reports,
                "query_summaries": query_rows,
                "parent_summaries": parent_rows,
            }
            layer_reports[str(layer_index)] = layer_report

        global_statistics: dict[str, Any] = {}
        for name, parts in global_metric_parts.items():
            count = sum(part["count"] for part in parts)
            mean = (
                sum(part["mean"] * part["count"] for part in parts) / count
                if count
                else 0.0
            )
            variance = (
                sum(
                    part["count"]
                    * (part["variance"] + (part["mean"] - mean) ** 2)
                    for part in parts
                )
                / count
                if count
                else 0.0
            )
            global_statistics[name] = {
                "count": count,
                "mean": mean,
                "variance": variance,
                "min": min((part["min"] for part in parts), default=0.0),
                "max": max((part["max"] for part in parts), default=0.0),
            }

        metrics = {
            name: stats["mean"] for name, stats in global_statistics.items()
        }
        metrics.update(
            {
                "gamma_query_variance": (
                    global_gamma_weighted / global_gamma_streams
                    if global_gamma_streams
                    else 0.0
                ),
                "gamma_query_variance_max": global_gamma_max,
                "posterior_top1_change_rate": (
                    top1_disagreements / top1_comparisons
                    if top1_comparisons
                    else 0.0
                ),
                "posterior_top1_any_change": any_top1_change,
            }
        )
        return {
            "schema_version": 1,
            "mode": self.mode,
            "metadata": deepcopy(self.metadata),
            "forward_count": max(
                (layer.forward_count for layer in self.layers.values()), default=0
            ),
            "metrics": metrics,
            "metric_statistics": global_statistics,
            "layers": layer_reports,
            "stores_raw_kv": False,
        }
