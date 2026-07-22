#!/usr/bin/env python3
"""Focused and immutable semantic-map checks for FPCT GPU R2l."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from types import SimpleNamespace
from typing import Any

import torch

from rosetta.model.fpct_attention import (
    FPCTSidecarSegment,
    bind_fpct_layout_layer_semantics,
    build_fpct_packed_layout,
    fpct_qwen_eager_attention_forward,
    fpct_qwen_hierarchical_attention_forward,
    pack_fpct_memory,
)
from script.analysis.fpct_gpu_r2k_diagnostic_verify import trace_summary
from script.experiment import fpct_gpu_r2_runner as r2


class R2lGateError(RuntimeError):
    pass


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def ulp_max(left: torch.Tensor, right: torch.Tensor) -> int:
    if left.dtype == torch.float32:
        left_bits = left.contiguous().view(torch.int32).to(torch.int64)
        right_bits = right.contiguous().view(torch.int32).to(torch.int64)
    elif left.dtype == torch.bfloat16:
        left_bits = left.contiguous().view(torch.int16).to(torch.int32)
        right_bits = right.contiguous().view(torch.int16).to(torch.int32)
    else:
        raise R2lGateError(f"unsupported ULP dtype {left.dtype}")
    return int((left_bits - right_bits).abs().max().cpu()) if left.numel() else 0


def exact_report(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    return {
        "equal": bool(torch.equal(left, right)),
        "left_sha256": tensor_sha256(left),
        "right_sha256": tensor_sha256(right),
        "max_abs": float((left.float() - right.float()).abs().max().cpu()),
        "ulp_max": ulp_max(left, right),
    }


def _segment(
    device: torch.device,
    *,
    dtype: torch.dtype,
    batch: int,
    hkv: int,
    equivalent: torch.Tensor | None,
    force_native: torch.Tensor | None = None,
) -> FPCTSidecarSegment:
    key = torch.zeros(batch, hkv, 3, 2, 8, device=device, dtype=dtype)
    value = torch.zeros_like(key)
    prior = torch.full((batch, 3, 2), 0.5, device=device, dtype=torch.float32)
    valid = torch.ones_like(prior, dtype=torch.bool)
    return FPCTSidecarSegment(
        2,
        key,
        value,
        prior,
        valid,
        max_slots_hint=9,
        source_length_hint=6,
        prior_sha256="r2l-synthetic-prior",
        certified=True,
        parent_force_native=force_native,
        parent_equivalent=equivalent,
    )


def _semantic_truth_checks(device: torch.device) -> dict[str, bool]:
    all_true = torch.ones(2, 3, device=device, dtype=torch.bool)
    layout = build_fpct_packed_layout(
        9, [_segment(device, dtype=torch.float32, batch=2, hkv=1, equivalent=all_true)]
    )
    bound = bind_fpct_layout_layer_semantics(
        layout,
        [(0, [_segment(device, dtype=torch.float32, batch=2, hkv=1, equivalent=all_true)])],
    )
    complete = bound.semantic_parent_equivalent(0)

    unknown_segment = _segment(
        device, dtype=torch.float32, batch=2, hkv=1, equivalent=None
    )
    unknown_layout = bind_fpct_layout_layer_semantics(
        build_fpct_packed_layout(9, [unknown_segment]), [(0, [unknown_segment])]
    )
    unknown = unknown_layout.semantic_parent_equivalent(0)
    expected_unknown = torch.ones(2, 9, device=device, dtype=torch.bool)
    expected_unknown[:, 2:5] = False
    return {
        "native_parent_map_complete": complete is not None and bool(complete.all()),
        "unknown_sidecar_fails_closed": unknown is not None and bool(torch.equal(unknown, expected_unknown)),
    }


def _mixed_tensor_case(
    device: torch.device, dtype: torch.dtype, hkv: int
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(20260722 + hkv)
    batch, source_length, dimension, hq = 2, 9, 8, hkv * 2
    query = torch.randn(
        batch, hq, 2, dimension, generator=generator, device=device, dtype=dtype
    )
    parent_key = torch.randn(
        batch, hkv, source_length, dimension,
        generator=generator, device=device, dtype=dtype,
    )
    parent_value = torch.randn(
        batch, hkv, source_length, dimension,
        generator=generator, device=device, dtype=dtype,
    )
    candidate_key = parent_key[:, :, 2:5, None, :].expand(-1, -1, -1, 2, -1).clone()
    candidate_value = parent_value[:, :, 2:5, None, :].expand(-1, -1, -1, 2, -1).clone()
    candidate_key[1, :, 1, 0] += 1.0
    candidate_value[1, :, 1, 0] -= 0.5
    prior = torch.full((batch, 3, 2), 0.5, device=device, dtype=torch.float32)
    valid = torch.ones_like(prior, dtype=torch.bool)
    equivalent = torch.tensor(
        [[True, True, True], [True, False, True]], device=device, dtype=torch.bool
    )
    segment = FPCTSidecarSegment(
        2,
        candidate_key,
        candidate_value,
        prior,
        valid,
        max_slots_hint=12,
        source_length_hint=9,
        prior_sha256="r2l-mixed-prior",
        certified=True,
        parent_equivalent=equivalent,
    )
    layout = bind_fpct_layout_layer_semantics(
        build_fpct_packed_layout(source_length, [segment]), [(0, [segment])]
    )
    semantic = layout.semantic_parent_equivalent(0)
    if semantic is None:
        raise R2lGateError("missing semantic map")
    mask = torch.zeros(batch, 1, query.shape[2], source_length, device=device)
    packed = pack_fpct_memory(
        parent_key,
        parent_value,
        mask,
        [segment],
        query_length=query.shape[2],
        layout=layout,
        semantic_parent_equivalent=semantic,
    )
    module = SimpleNamespace(num_key_value_groups=hq // hkv, training=False)
    parent_output, _ = fpct_qwen_eager_attention_forward(
        module, query, parent_key, parent_value, mask, scaling=1 / math.sqrt(dimension)
    )
    factorized_output, _ = fpct_qwen_hierarchical_attention_forward(
        module,
        query,
        packed,
        parent_key,
        parent_value,
        mask,
        scaling=1 / math.sqrt(dimension),
    )
    exact = exact_report(parent_output[0], factorized_output[0])
    return {
        "dtype": str(dtype),
        "hkv": hkv,
        "all_parent_equivalent": packed.all_parent_equivalent.detach().cpu().tolist(),
        "exact_sample": exact,
        "active_sample_max_abs": float(
            (parent_output[1].float() - factorized_output[1].float()).abs().max().cpu()
        ),
    }


def synthetic(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise R2lGateError("CUDA unavailable")
    output = Path(args.output)
    if output.exists():
        raise R2lGateError("refusing to overwrite synthetic output")
    device = torch.device("cuda:0")
    truth = _semantic_truth_checks(device)
    rows = [
        _mixed_tensor_case(device, dtype, hkv)
        for dtype in (torch.float32, torch.bfloat16)
        for hkv in (1, 2)
    ]
    checks = {
        **truth,
        "mixed_memory_exact_null_bitwise": all(
            row["exact_sample"]["equal"]
            and row["exact_sample"]["max_abs"] == 0
            and row["exact_sample"]["ulp_max"] == 0
            for row in rows
        ),
        "mixed_batch_exact_active_isolation": all(
            row["all_parent_equivalent"] == [True, False]
            and row["active_sample_max_abs"] > 0
            for row in rows
        ),
    }
    payload = {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2l_synthetic_mixed_memory_v1",
        "status": "GO" if all(checks.values()) else "DIAGNOSTIC_FAILED",
        "checks": checks,
        "rows": rows,
        "device": torch.cuda.get_device_name(0),
        "accuracy_or_correctness_accessed": False,
    }
    atomic_json(output, payload)
    if payload["status"] != "GO":
        raise R2lGateError(f"synthetic semantic checks failed: {checks}")
    return payload


def _compact_exact(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    left = r2._compact_index(left_rows)
    right = r2._compact_index(right_rows)
    fields = ("pre_o_proj_last", "post_o_proj_last", "residual_last")
    reports = []
    all_equal = set(left) == set(right)
    for panel_id in sorted(set(left) & set(right)):
        all_equal = all_equal and set(left[panel_id]) == set(right[panel_id])
        for layer in sorted(set(left[panel_id]) & set(right[panel_id]), key=int):
            for field in fields:
                if field not in left[panel_id][layer] or field not in right[panel_id][layer]:
                    continue
                report = exact_report(
                    left[panel_id][layer][field], right[panel_id][layer][field]
                )
                reports.append({"panel_id": panel_id, "layer": int(layer), "field": field, **report})
                all_equal = all_equal and report["equal"] and report["ulp_max"] == 0
    first = next((row for row in reports if not row["equal"]), None)
    return {
        "all_equal": all_equal,
        "comparison_count": len(reports),
        "first_divergence": first,
        "records_sha256": hashlib.sha256(
            json.dumps(reports, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if output.exists():
        raise R2lGateError("refusing to overwrite focused aggregate")
    conditions_root = Path(args.conditions_root)
    floors = r2.load_json(Path(args.floors))
    synthetic_result = r2.load_json(Path(args.synthetic))
    exact_by_dtype = {}
    forced_by_dtype = {}
    precollapse = {}
    for dtype_name in ("fp32", "bf16"):
        loaded = {
            condition: r2._load_condition(conditions_root, dtype_name, condition)
            for condition in (
                "OP01_CPOST_NATIVE",
                "OP02_F_NATIVE",
                "OP03_F_REP_NATIVE",
                "OP04_F_FORCED",
                "OP05_F_REP_FORCED",
            )
        }
        cpost, native, replicated = (
            loaded["OP01_CPOST_NATIVE"],
            loaded["OP02_F_NATIVE"],
            loaded["OP03_F_REP_NATIVE"],
        )
        exact_by_dtype[dtype_name] = {
            "cpost_vs_f_logits": exact_report(cpost[1], native[1]),
            "f_vs_replicated_logits": exact_report(native[1], replicated[1]),
            "cpost_vs_f_layers": _compact_exact(cpost[3], native[3]),
            "f_vs_replicated_layers": _compact_exact(native[3], replicated[3]),
        }
        precollapse[dtype_name] = r2._precollapse_identity(cpost[3], native[3])
        forced = loaded["OP04_F_FORCED"]
        forced_rep = loaded["OP05_F_REP_FORCED"]
        tau = {key: float(value["tau"]) for key, value in floors["floors"].items()}
        metrics = {
            name: r2._metric_from_trace(forced[2], name)
            for name in ("d_k", "d_v", "candidate_logit_range")
        }
        forced_by_dtype[dtype_name] = {
            "delta_fact_max_abs": r2._max_abs(forced[1], forced_rep[1]),
            "metrics": {name: max((value for _layer, value in rows), default=0.0) for name, rows in metrics.items()},
            "active": (
                r2._max_abs(forced[1], forced_rep[1]) > tau["delta_fact_max_abs"]
                and all(
                    max((value for _layer, value in rows), default=0.0) > tau[name]
                    for name, rows in metrics.items()
                )
            ),
        }

    block = r2.load_json(Path(args.latency_block))
    resource = {}
    peak_hbm = []
    expansion = []
    for label in ("checkpoint_native", "forced_on"):
        arms = {row["operator"]: row for row in block["canaries"][label]["arms"]}
        cpost_timing = arms["c_post"]["timing"]
        f_timing = arms["f"]["timing"]
        cpost_cuda = [float(value) for value in cpost_timing["cuda_event_seconds"]]
        f_cuda = [float(value) for value in f_timing["cuda_event_seconds"]]
        median_ratio = statistics.median(f_cuda) / statistics.median(cpost_cuda)
        p95_ratio = _percentile(f_cuda, 0.95) / _percentile(cpost_cuda, 0.95)
        resource[label] = {
            "cuda_median_ratio": median_ratio,
            "cuda_p95_ratio": p95_ratio,
            "pass": median_ratio <= 1.5 and p95_ratio <= 1.75,
        }
        for arm in arms.values():
            peak_hbm.append(float(arm["peak_hbm_gib"]))
            expansion.extend(float(value) for value in arm["expanded_slot_ratios"])

    traces = {
        name: trace_summary(Path(args.trace_root) / f"{name}.chrome.json")
        for name in ("c_post", "f")
    }
    exact_ok = all(
        row["cpost_vs_f_logits"]["equal"]
        and row["f_vs_replicated_logits"]["equal"]
        and row["cpost_vs_f_layers"]["all_equal"]
        and row["f_vs_replicated_layers"]["all_equal"]
        for row in exact_by_dtype.values()
    )
    checks = {
        "native_parent_map_complete": bool(synthetic_result["checks"]["native_parent_map_complete"]),
        "unknown_sidecar_fails_closed": bool(synthetic_result["checks"]["unknown_sidecar_fails_closed"]),
        "mixed_memory_exact_null_bitwise": exact_ok and bool(synthetic_result["checks"]["mixed_memory_exact_null_bitwise"]),
        "mixed_batch_exact_active_isolation": bool(synthetic_result["checks"]["mixed_batch_exact_active_isolation"]),
        "actual_qwen_decode4_exact_null": exact_ok,
        "active_route_not_bypassed": all(row["active"] for row in forced_by_dtype.values()),
        "precollapse_candidate_identity": all(precollapse.values()),
        "focused_resource_pass": all(row["pass"] for row in resource.values()),
        "hot_path_no_sync": not traces["c_post"]["hot_path_sync_events"] and not traces["f"]["hot_path_sync_events"],
    }
    payload = {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2l_focused_diagnostic_result_v1",
        "classification": "DIAGNOSTIC_ONLY",
        "status": "DIAGNOSTIC_QUALIFIED" if all(checks.values()) else "DIAGNOSTIC_FAILED",
        "checks": checks,
        "exact_by_dtype": exact_by_dtype,
        "forced_by_dtype": forced_by_dtype,
        "precollapse_candidate_identity": precollapse,
        "resource": {
            "canaries": resource,
            "peak_hbm_gib_max": max(peak_hbm),
            "expansion_mean": statistics.mean(expansion),
            "expansion_p95": _percentile(expansion, 0.95),
        },
        "traces": traces,
        "accuracy_or_correctness_accessed": False,
        "may_produce_r2l_go": False,
    }
    atomic_json(output, payload)
    if payload["status"] != "DIAGNOSTIC_QUALIFIED":
        raise R2lGateError(f"focused diagnostic failed: {checks}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic_parser = commands.add_parser("synthetic")
    synthetic_parser.add_argument("--output", required=True)
    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--conditions-root", required=True)
    aggregate_parser.add_argument("--floors", required=True)
    aggregate_parser.add_argument("--synthetic", required=True)
    aggregate_parser.add_argument("--latency-block", required=True)
    aggregate_parser.add_argument("--trace-root", required=True)
    aggregate_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = synthetic(args) if args.command == "synthetic" else aggregate(args)
    print(json.dumps({"status": payload["status"], "classification": payload.get("classification")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
