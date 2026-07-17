"""Streaming diagnostics for token/head alignment confidence gates.

The projector records final key/value confidence tensors with shape ``[B, H, N, 1]``.
This module reduces those tensors online so evaluation never needs to retain or write
the raw gate tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch


GATE_DIAGNOSTICS_SCHEMA_VERSION = 1
GATE_DEFINITION = (
    "sigmoid(logit(source_confidence) + clamp(key_or_value_bias + "
    "entropy_scale*source_entropy + layer_scale*token_head_mlp(features), "
    "+/-alignment_confidence_max_delta))"
)
_DIAGNOSTIC_TENSOR_ATTRIBUTES = (
    "last_alignment_key_delta_tensor",
    "last_alignment_value_delta_tensor",
    "last_alignment_key_confidence_tensor",
    "last_alignment_value_confidence_tensor",
    "last_alignment_source_confidence_tensor",
    "last_alignment_entropy_tensor",
    "last_alignment_regularization_mask_tensor",
)
_COMPACT_GATE_ATTRIBUTES = (
    "last_alignment_key_confidence",
    "last_alignment_value_confidence",
    "last_alignment_key_confidence_std",
    "last_alignment_value_confidence_std",
)


@dataclass
class RunningGateStats:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    saturation_low_count: int = 0
    saturation_high_count: int = 0

    def update(
        self,
        values: torch.Tensor,
        *,
        low_threshold: float,
        high_threshold: float,
    ) -> None:
        detached = values.detach().float().reshape(-1)
        if detached.numel() == 0:
            return
        finite = detached[torch.isfinite(detached)]
        if finite.numel() == 0:
            return
        count = int(finite.numel())
        current_min = float(finite.min().item())
        current_max = float(finite.max().item())
        self.count += count
        self.total += float(finite.sum().item())
        self.total_square += float(finite.square().sum().item())
        self.minimum = (
            current_min if self.minimum is None else min(self.minimum, current_min)
        )
        self.maximum = (
            current_max if self.maximum is None else max(self.maximum, current_max)
        )
        self.saturation_low_count += int((finite <= low_threshold).sum().item())
        self.saturation_high_count += int((finite >= high_threshold).sum().item())

    def merge(self, other: "RunningGateStats") -> None:
        if other.count <= 0:
            return
        self.count += other.count
        self.total += other.total
        self.total_square += other.total_square
        self.minimum = (
            other.minimum
            if self.minimum is None
            else (
                self.minimum
                if other.minimum is None
                else min(self.minimum, other.minimum)
            )
        )
        self.maximum = (
            other.maximum
            if self.maximum is None
            else (
                self.maximum
                if other.maximum is None
                else max(self.maximum, other.maximum)
            )
        )
        self.saturation_low_count += other.saturation_low_count
        self.saturation_high_count += other.saturation_high_count

    def state_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "total": self.total,
            "total_square": self.total_square,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "saturation_low_count": self.saturation_low_count,
            "saturation_high_count": self.saturation_high_count,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RunningGateStats":
        return cls(
            count=int(state.get("count", 0)),
            total=float(state.get("total", 0.0)),
            total_square=float(state.get("total_square", 0.0)),
            minimum=(
                None
                if state.get("minimum") is None
                else float(state.get("minimum"))
            ),
            maximum=(
                None
                if state.get("maximum") is None
                else float(state.get("maximum"))
            ),
            saturation_low_count=int(state.get("saturation_low_count", 0)),
            saturation_high_count=int(state.get("saturation_high_count", 0)),
        )

    def finalize(self) -> Dict[str, Any]:
        if self.count <= 0:
            return {
                "count": 0,
                "mean": None,
                "variance": None,
                "std": None,
                "minimum": None,
                "maximum": None,
                "saturation_low_rate": None,
                "saturation_high_rate": None,
            }
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "variance": variance,
            "std": math.sqrt(variance),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "saturation_low_rate": self.saturation_low_count / self.count,
            "saturation_high_rate": self.saturation_high_count / self.count,
        }


def _layer_stage(layer: int, num_layers: int) -> str:
    if num_layers <= 0:
        return "unknown"
    # Equivalent to numpy.array_split(range(num_layers), 3), without numpy.
    early_size = (num_layers + 2) // 3
    middle_size = (num_layers + 1) // 3
    if layer < early_size:
        return "early"
    if layer < early_size + middle_size:
        return "middle"
    return "late"


def _layer_ranges(num_layers: int) -> Dict[str, Dict[str, Optional[int]]]:
    output: Dict[str, Dict[str, Optional[int]]] = {}
    for stage in ("early", "middle", "late"):
        layers = [index for index in range(num_layers) if _layer_stage(index, num_layers) == stage]
        output[stage] = {
            "start": min(layers) if layers else None,
            "end": max(layers) if layers else None,
            "count": len(layers),
        }
    return output


def clear_projector_gate_diagnostic_records(projectors: Iterable[Any]) -> None:
    """Release projector diagnostic tensors after every evaluated example."""
    for projector in projectors:
        records = getattr(projector, "alignment_diagnostic_records", None)
        if isinstance(records, list):
            records.clear()
        for attribute in _DIAGNOSTIC_TENSOR_ATTRIBUTES:
            if hasattr(projector, attribute):
                setattr(projector, attribute, None)


def clear_projector_compact_gate_values(projectors: Iterable[Any]) -> None:
    """Prevent a skipped projector call from reusing the preceding example's values."""
    for projector in projectors:
        for attribute in _COMPACT_GATE_ATTRIBUTES:
            if hasattr(projector, attribute):
                setattr(projector, attribute, None)


def _target_layer_by_projector(model: Any, projector_count: int) -> list[int]:
    mapping = list(range(projector_count))
    projector_dict = getattr(model, "projector_dict", None)
    if not isinstance(projector_dict, Mapping):
        return mapping
    for target_sources in projector_dict.values():
        if not isinstance(target_sources, Mapping):
            continue
        for target_layers in target_sources.values():
            if not isinstance(target_layers, Mapping):
                continue
            for target_layer, entries in target_layers.items():
                try:
                    target_layer_index = int(target_layer)
                except (TypeError, ValueError):
                    continue
                if isinstance(entries, tuple):
                    entries = [entries]
                if not isinstance(entries, Sequence):
                    continue
                for entry in entries:
                    if not isinstance(entry, Sequence) or len(entry) != 2:
                        continue
                    try:
                        projector_index = int(entry[1])
                    except (TypeError, ValueError):
                        continue
                    if 0 <= projector_index < projector_count:
                        mapping[projector_index] = target_layer_index
    return mapping


def configure_projector_gate_diagnostics(model: Any, enabled: bool) -> Dict[str, Any]:
    """Enable capture only for token/head gates, leaving non-gated controls empty."""
    projectors = list(getattr(model, "projector_list", []) or [])
    gate_projectors = 0
    for projector in projectors:
        is_token_head_gate = (
            getattr(projector, "alignment_confidence_gate_mode", "none")
            == "token_mlp"
        )
        projector.capture_alignment_diagnostics = bool(enabled and is_token_head_gate)
        gate_projectors += int(is_token_head_gate)
    clear_projector_gate_diagnostic_records(projectors)
    return {
        "projector_count": len(projectors),
        "gate_projector_count": gate_projectors,
        "target_layer_by_projector": _target_layer_by_projector(
            model, len(projectors)
        ),
    }


class GateDiagnosticsAccumulator:
    """Online aggregate of final key/value token/head confidence gates."""

    def __init__(
        self,
        *,
        low_threshold: float = 0.05,
        high_threshold: float = 0.95,
        relative_token_bins: int = 10,
    ) -> None:
        if not 0.0 <= low_threshold < high_threshold <= 1.0:
            raise ValueError(
                "gate saturation thresholds must satisfy "
                f"0 <= low < high <= 1, got {low_threshold}, {high_threshold}"
            )
        if relative_token_bins <= 0:
            raise ValueError("relative_token_bins must be positive")
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.relative_token_bins = int(relative_token_bins)
        self.num_layers = 0
        self.examples_seen = 0
        self.examples_with_gate = 0
        self.record_count = 0
        self.projector_count = 0
        self.gate_projector_count = 0
        self.target_layer_by_projector: list[int] = []
        self.compact_only = False
        self._stats: MutableMapping[Tuple[str, Tuple[Any, ...], str], RunningGateStats] = {}

    def note_projectors(
        self,
        projector_count: int,
        gate_projector_count: int,
        target_layer_by_projector: Optional[Sequence[int]] = None,
    ) -> None:
        self.projector_count = max(self.projector_count, int(projector_count))
        self.gate_projector_count = max(
            self.gate_projector_count, int(gate_projector_count)
        )
        if target_layer_by_projector is not None:
            candidate = [int(value) for value in target_layer_by_projector]
            if not self.target_layer_by_projector:
                self.target_layer_by_projector = candidate
            elif self.target_layer_by_projector != candidate:
                raise ValueError("inconsistent projector-to-target-layer mapping")
        self.num_layers = max(
            self.num_layers,
            max(self.target_layer_by_projector, default=int(projector_count) - 1) + 1,
        )

    def _entry(
        self, axis: str, coordinates: Sequence[Any], kv: str
    ) -> RunningGateStats:
        key = (axis, tuple(coordinates), kv)
        if key not in self._stats:
            self._stats[key] = RunningGateStats()
        return self._stats[key]

    def _update(
        self,
        axis: str,
        coordinates: Sequence[Any],
        kv: str,
        values: torch.Tensor,
    ) -> None:
        self._entry(axis, coordinates, kv).update(
            values,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )

    def consume_projectors(self, projectors: Iterable[Any]) -> Dict[str, Any]:
        """Consume one example's records and always clear the projector buffers."""
        projector_list = list(projectors)
        self.examples_seen += 1
        self.note_projectors(
            len(projector_list),
            sum(
                getattr(projector, "alignment_confidence_gate_mode", "none")
                == "token_mlp"
                for projector in projector_list
            ),
        )
        example_key = RunningGateStats()
        example_value = RunningGateStats()
        example_records = 0
        try:
            for projector_index, projector in enumerate(projector_list):
                if (
                    getattr(projector, "alignment_confidence_gate_mode", "none")
                    != "token_mlp"
                ):
                    continue
                layer = (
                    self.target_layer_by_projector[projector_index]
                    if projector_index < len(self.target_layer_by_projector)
                    else projector_index
                )
                for record in list(
                    getattr(projector, "alignment_diagnostic_records", []) or []
                ):
                    key_gate = record.get("key_confidence")
                    value_gate = record.get("value_confidence")
                    if not isinstance(key_gate, torch.Tensor) or not isinstance(
                        value_gate, torch.Tensor
                    ):
                        continue
                    if key_gate.ndim != 4 or value_gate.shape != key_gate.shape:
                        continue
                    batch, heads, tokens, trailing = key_gate.shape
                    if trailing != 1 or batch <= 0 or heads <= 0 or tokens <= 0:
                        continue
                    example_records += 1
                    self.record_count += 1
                    stage = _layer_stage(layer, self.num_layers)
                    for kv, gate, example_stats in (
                        ("key", key_gate, example_key),
                        ("value", value_gate, example_value),
                    ):
                        example_stats.update(
                            gate,
                            low_threshold=self.low_threshold,
                            high_threshold=self.high_threshold,
                        )
                        self._update("global", (), kv, gate)
                        self._update("layer", (layer,), kv, gate)
                        self._update("stage", (stage,), kv, gate)
                        for head in range(heads):
                            self._update(
                                "layer_head", (layer, head), kv, gate[:, head]
                            )
                        for token in range(tokens):
                            token_values = gate[:, :, token]
                            self._update(
                                "token_position", (token,), kv, token_values
                            )
                            relative_bin = min(
                                self.relative_token_bins - 1,
                                (token * self.relative_token_bins) // tokens,
                            )
                            self._update(
                                "relative_token_bin",
                                (relative_bin,),
                                kv,
                                token_values,
                            )
        finally:
            clear_projector_gate_diagnostic_records(projector_list)

        if example_records <= 0:
            return {
                "gate_diagnostics_status": "unavailable",
                "gate_record_count": 0,
                "gate_token_count": 0,
            }
        self.examples_with_gate += 1
        combined = RunningGateStats()
        combined.merge(example_key)
        combined.merge(example_value)
        key_summary = example_key.finalize()
        value_summary = example_value.finalize()
        combined_summary = combined.finalize()
        return {
            "gate_diagnostics_status": "ok",
            "gate_record_count": example_records,
            "gate_token_count": key_summary["count"],
            "gate": combined_summary["mean"],
            "key_gate_mean": key_summary["mean"],
            "key_gate_std": key_summary["std"],
            "key_gate_saturation_low_rate": key_summary["saturation_low_rate"],
            "key_gate_saturation_high_rate": key_summary[
                "saturation_high_rate"
            ],
            "value_gate_mean": value_summary["mean"],
            "value_gate_std": value_summary["std"],
            "value_gate_saturation_low_rate": value_summary[
                "saturation_low_rate"
            ],
            "value_gate_saturation_high_rate": value_summary[
                "saturation_high_rate"
            ],
        }

    def consume_compact_projectors(self, projectors: Iterable[Any]) -> Dict[str, Any]:
        """Consume projector scalar means without enabling raw tensor capture.

        This is intended for full benchmark evaluation. Detailed head/token statistics
        are collected separately by the single-GPU post-hoc diagnostic action.
        """
        projector_list = list(projectors)
        self.compact_only = True
        self.examples_seen += 1
        self.note_projectors(
            len(projector_list),
            sum(
                getattr(projector, "alignment_confidence_gate_mode", "none")
                == "token_mlp"
                for projector in projector_list
            ),
        )
        key_means: list[float] = []
        value_means: list[float] = []
        try:
            for projector_index, projector in enumerate(projector_list):
                if (
                    getattr(projector, "alignment_confidence_gate_mode", "none")
                    != "token_mlp"
                ):
                    continue
                key_value = getattr(projector, "last_alignment_key_confidence", None)
                value_value = getattr(
                    projector, "last_alignment_value_confidence", None
                )
                if key_value is None or value_value is None:
                    continue
                key_mean = float(
                    key_value.item()
                    if isinstance(key_value, torch.Tensor)
                    else key_value
                )
                value_mean = float(
                    value_value.item()
                    if isinstance(value_value, torch.Tensor)
                    else value_value
                )
                if not math.isfinite(key_mean) or not math.isfinite(value_mean):
                    continue
                layer = (
                    self.target_layer_by_projector[projector_index]
                    if projector_index < len(self.target_layer_by_projector)
                    else projector_index
                )
                stage = _layer_stage(layer, self.num_layers)
                key_tensor = torch.tensor([key_mean])
                value_tensor = torch.tensor([value_mean])
                for kv, value_tensor_item in (
                    ("key", key_tensor),
                    ("value", value_tensor),
                ):
                    self._update("global", (), kv, value_tensor_item)
                    self._update("layer", (layer,), kv, value_tensor_item)
                    self._update("stage", (stage,), kv, value_tensor_item)
                key_means.append(key_mean)
                value_means.append(value_mean)
                self.record_count += 1
        finally:
            clear_projector_compact_gate_values(projector_list)

        if not key_means:
            return {
                "gate_diagnostics_status": "unavailable",
                "gate_record_count": 0,
                "gate_token_count": 0,
            }
        self.examples_with_gate += 1
        key_mean = sum(key_means) / len(key_means)
        value_mean = sum(value_means) / len(value_means)
        key_variance = sum((value - key_mean) ** 2 for value in key_means) / len(
            key_means
        )
        value_variance = sum(
            (value - value_mean) ** 2 for value in value_means
        ) / len(value_means)
        return {
            "gate_diagnostics_status": "compact",
            "gate_record_count": len(key_means),
            "gate_token_count": None,
            "gate": 0.5 * (key_mean + value_mean),
            "key_gate_mean": key_mean,
            "key_gate_std": math.sqrt(max(key_variance, 0.0)),
            "key_gate_saturation_low_rate": None,
            "key_gate_saturation_high_rate": None,
            "value_gate_mean": value_mean,
            "value_gate_std": math.sqrt(max(value_variance, 0.0)),
            "value_gate_saturation_low_rate": None,
            "value_gate_saturation_high_rate": None,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": GATE_DIAGNOSTICS_SCHEMA_VERSION,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "relative_token_bins": self.relative_token_bins,
            "num_layers": self.num_layers,
            "examples_seen": self.examples_seen,
            "examples_with_gate": self.examples_with_gate,
            "record_count": self.record_count,
            "projector_count": self.projector_count,
            "gate_projector_count": self.gate_projector_count,
            "target_layer_by_projector": self.target_layer_by_projector,
            "compact_only": self.compact_only,
            "stats": [
                {
                    "axis": axis,
                    "coordinates": list(coordinates),
                    "kv": kv,
                    "moments": stats.state_dict(),
                }
                for (axis, coordinates, kv), stats in sorted(
                    self._stats.items(), key=lambda item: str(item[0])
                )
            ],
        }

    def merge_state_dict(self, state: Mapping[str, Any]) -> None:
        if float(state.get("low_threshold", self.low_threshold)) != self.low_threshold:
            raise ValueError("cannot merge gate diagnostics with different low thresholds")
        if float(state.get("high_threshold", self.high_threshold)) != self.high_threshold:
            raise ValueError(
                "cannot merge gate diagnostics with different high thresholds"
            )
        if int(state.get("relative_token_bins", self.relative_token_bins)) != self.relative_token_bins:
            raise ValueError(
                "cannot merge gate diagnostics with different relative token bins"
            )
        self.num_layers = max(self.num_layers, int(state.get("num_layers", 0)))
        self.examples_seen += int(state.get("examples_seen", 0))
        self.examples_with_gate += int(state.get("examples_with_gate", 0))
        self.record_count += int(state.get("record_count", 0))
        self.projector_count = max(
            self.projector_count, int(state.get("projector_count", 0))
        )
        self.gate_projector_count = max(
            self.gate_projector_count, int(state.get("gate_projector_count", 0))
        )
        self.compact_only = self.compact_only or bool(state.get("compact_only", False))
        candidate_mapping = [
            int(value) for value in state.get("target_layer_by_projector", [])
        ]
        if candidate_mapping:
            if not self.target_layer_by_projector:
                self.target_layer_by_projector = candidate_mapping
            elif self.target_layer_by_projector != candidate_mapping:
                raise ValueError("inconsistent projector-to-target-layer mapping")
        for row in state.get("stats", []):
            stats = self._entry(
                str(row["axis"]), tuple(row.get("coordinates", [])), str(row["kv"])
            )
            stats.merge(RunningGateStats.from_state_dict(row.get("moments", {})))

    def _paired_rows(self, axis: str) -> list[Dict[str, Any]]:
        coordinates = sorted(
            {
                key_coordinates
                for key_axis, key_coordinates, _ in self._stats
                if key_axis == axis
            },
            key=str,
        )
        rows: list[Dict[str, Any]] = []
        coordinate_names = {
            "layer": ("layer",),
            "stage": ("stage",),
            "layer_head": ("layer", "head"),
            "token_position": ("token_position",),
            "relative_token_bin": ("relative_token_bin",),
        }[axis]
        for coordinate in coordinates:
            row = dict(zip(coordinate_names, coordinate))
            if "layer" in row:
                row["stage"] = _layer_stage(int(row["layer"]), self.num_layers)
            row["key"] = self._entry(axis, coordinate, "key").finalize()
            row["value"] = self._entry(axis, coordinate, "value").finalize()
            rows.append(row)
        return rows

    def finalize(self, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        key_global = self._entry("global", (), "key").finalize()
        value_global = self._entry("global", (), "value").finalize()
        combined = RunningGateStats()
        combined.merge(self._entry("global", (), "key"))
        combined.merge(self._entry("global", (), "value"))
        status = (
            "compact"
            if combined.count > 0 and self.compact_only
            else ("ok" if combined.count > 0 else "unavailable")
        )
        heads_by_layer: Dict[str, int] = {}
        for axis, coordinates, _ in self._stats:
            if axis != "layer_head" or len(coordinates) != 2:
                continue
            layer, head = int(coordinates[0]), int(coordinates[1])
            heads_by_layer[str(layer)] = max(heads_by_layer.get(str(layer), 0), head + 1)
        artifact: Dict[str, Any] = {
            "schema_version": GATE_DIAGNOSTICS_SCHEMA_VERSION,
            "status": status,
            "unavailable_reason": (
                None if status in {"ok", "compact"} else "no_token_head_gate_records"
            ),
            "gate_definition": GATE_DEFINITION,
            "layer_axis": "target_model_layer",
            "projector_to_target_layer": self.target_layer_by_projector,
            "saturation_thresholds": {
                "low": self.low_threshold,
                "high": self.high_threshold,
                "low_rule": f"gate <= {self.low_threshold}",
                "high_rule": f"gate >= {self.high_threshold}",
            },
            "token_axis": {
                "token_position": "section-local zero-based token position",
                "relative_token_bin": (
                    f"section-local relative position split into "
                    f"{self.relative_token_bins} equal-width bins"
                ),
            },
            "aggregation_scope": (
                "per-example projector means; use the checkpoint post-hoc artifact "
                "for exact head/token distributions"
                if status == "compact"
                else "final token/head gate values"
            ),
            "layer_groups": _layer_ranges(self.num_layers),
            "dimensions": {
                "num_target_layers": self.num_layers,
                "heads_per_layer": heads_by_layer,
                "relative_token_bins": self.relative_token_bins,
            },
            "counts": {
                "examples_seen": self.examples_seen,
                "examples_with_gate": self.examples_with_gate,
                "records": self.record_count,
                "projectors": self.projector_count,
                "token_head_gate_projectors": self.gate_projector_count,
            },
            "global": {
                "combined": combined.finalize(),
                "key": key_global,
                "value": value_global,
            },
            "by_layer": self._paired_rows("layer"),
            "by_stage": self._paired_rows("stage"),
            "by_layer_head": self._paired_rows("layer_head"),
            "by_token_position": self._paired_rows("token_position"),
            "by_relative_token_bin": self._paired_rows("relative_token_bin"),
        }
        if metadata:
            artifact["metadata"] = dict(metadata)
        return artifact


def merge_gate_diagnostic_states(
    states: Iterable[Mapping[str, Any]],
    *,
    low_threshold: float = 0.05,
    high_threshold: float = 0.95,
    relative_token_bins: int = 10,
) -> GateDiagnosticsAccumulator:
    accumulator = GateDiagnosticsAccumulator(
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        relative_token_bins=relative_token_bins,
    )
    for state in states:
        accumulator.merge_state_dict(state)
    return accumulator
