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
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

from rosetta.model.fpct_attention import (
    FPCTSidecarSegment,
    bind_fpct_layout_layer_semantics,
    build_fpct_packed_layout,
    fpct_qwen_eager_attention_forward,
    fpct_qwen_hierarchical_attention_forward,
    pack_fpct_memory,
)
from rosetta.model.wrapper import RosettaModel
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


def _qwen_config(*, layers: int, kv_heads: int) -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    return config


def _qwen_model(
    *,
    layers: int,
    kv_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    state: dict[str, torch.Tensor] | None = None,
) -> Qwen3ForCausalLM:
    torch.manual_seed(20260722)
    model = Qwen3ForCausalLM(_qwen_config(layers=layers, kv_heads=kv_heads))
    if state is not None:
        model.load_state_dict(state)
    return model.to(device=device, dtype=dtype).eval()


def _base_forward(
    wrapper: RosettaModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: DynamicCache | None,
):
    return wrapper._base_model_forward_with_fpct(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache or DynamicCache(),
        use_cache=True,
        return_dict=True,
    )


def _capture_hidden(wrapper: RosettaModel) -> tuple[dict[int, list[torch.Tensor]], list[Any]]:
    hidden: dict[int, list[torch.Tensor]] = {}
    handles = []
    for index, layer in enumerate(wrapper.model_list[0].model.layers):
        def hook(_module, _inputs, output, *, layer_index=index):
            value = output[0] if isinstance(output, tuple) else output
            hidden.setdefault(layer_index, []).append(value.detach().clone())

        handles.append(layer.register_forward_hook(hook))
    return hidden, handles


def _cache_exact(left: DynamicCache, right: DynamicCache) -> dict[str, Any]:
    if len(left.key_cache) != len(right.key_cache):
        return {"equal": False, "layer_count": 0, "first_divergence": "layer_count"}
    comparisons = 0
    first = None
    for layer, (left_key, right_key, left_value, right_value) in enumerate(
        zip(left.key_cache, right.key_cache, left.value_cache, right.value_cache)
    ):
        for name, left_tensor, right_tensor in (
            ("key", left_key, right_key),
            ("value", left_value, right_value),
        ):
            report = exact_report(left_tensor, right_tensor)
            comparisons += 1
            if first is None and not (
                report["equal"] and report["max_abs"] == 0 and report["ulp_max"] == 0
            ):
                first = {"layer": layer, "field": name, **report}
    return {
        "equal": first is None,
        "layer_count": len(left.key_cache),
        "comparison_count": comparisons,
        "first_divergence": first,
    }


def _store_cache_exact_sidecars(
    wrapper: RosettaModel,
    cache: DynamicCache,
    *,
    layers: int,
    batch: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    prior = torch.full((batch, 3, 2), 0.5, device=cache.key_cache[0].device, dtype=torch.float32)
    valid = torch.ones_like(prior, dtype=torch.bool)
    equivalent = torch.ones(batch, 3, device=prior.device, dtype=torch.bool)
    candidate_checks = []
    for layer in range(layers):
        parent_key = cache.key_cache[layer]
        parent_value = cache.value_cache[layer]
        key = parent_key[:, :, 2:5, None, :].expand(-1, -1, -1, 2, -1).clone()
        value = parent_value[:, :, 2:5, None, :].expand(-1, -1, -1, 2, -1).clone()
        expected_key = parent_key[:, :, 2:5, None, :].expand_as(key)
        expected_value = parent_value[:, :, 2:5, None, :].expand_as(value)
        candidate_checks.append(
            exact_report(key, expected_key)["equal"]
            and exact_report(value, expected_value)["equal"]
        )
        wrapper._store_fpct_sidecar(
            layer,
            2,
            key.to(dtype=dtype),
            value.to(dtype=dtype),
            prior,
            valid,
            parent_equivalent=equivalent,
            prior_sha256="r2l-immutable-qwen-canonical-prior",
            max_slots_hint=9,
            source_length_hint=6,
            certified=True,
        )
    return {
        "candidate_kv_parent_bitwise": all(candidate_checks),
        "layer_count": layers,
    }


@torch.no_grad()
def run_actual_qwen_decode4_case(
    *,
    device: torch.device,
    dtype: torch.dtype,
    layers: int = 28,
    kv_heads: int = 2,
    decode_steps: int = 4,
) -> dict[str, Any]:
    batch = 2
    base = _qwen_model(
        layers=layers, kv_heads=kv_heads, dtype=dtype, device=torch.device("cpu")
    )
    state = {name: value.detach().cpu().clone() for name, value in base.state_dict().items()}
    del base
    cpost = RosettaModel(
        [_qwen_model(layers=layers, kv_heads=kv_heads, dtype=dtype, device=device, state=state)],
        fpct_operator="c_post",
        fpct_trace=True,
    )
    factorized = RosettaModel(
        [_qwen_model(layers=layers, kv_heads=kv_heads, dtype=dtype, device=device, state=state)],
        fpct_operator="f",
        fpct_trace=True,
    )
    if set(cpost.state_dict()) != set(factorized.state_dict()):
        raise R2lGateError("C_post/F Qwen parameter keys differ")
    cpost_hidden, cpost_handles = _capture_hidden(cpost)
    factorized_hidden, factorized_handles = _capture_hidden(factorized)
    step_rows = []
    semantic_complete = True
    candidate_identity = None
    try:
        input_ids = torch.tensor(
            [[4, 5, 6, 7, 8, 9], [0, 0, 10, 11, 12, 13]], device=device
        )
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]],
            device=device,
            dtype=torch.long,
        )
        cpost_output = _base_forward(cpost, input_ids, attention_mask, None)
        factorized_output = _base_forward(factorized, input_ids, attention_mask, None)
        prefill_logits = exact_report(cpost_output.logits, factorized_output.logits)
        prefill_cache = _cache_exact(
            cpost_output.past_key_values, factorized_output.past_key_values
        )
        if not prefill_logits["equal"] or not prefill_cache["equal"]:
            raise R2lGateError("Qwen prefill baseline diverged before sidecar binding")
        candidate_identity = _store_cache_exact_sidecars(
            factorized,
            factorized_output.past_key_values,
            layers=layers,
            batch=batch,
            dtype=dtype,
        )
        cpost_cache = cpost_output.past_key_values
        factorized_cache = factorized_output.past_key_values
        for step in range(decode_steps):
            input_ids = torch.tensor([[14 + step], [19 + step]], device=device)
            attention_mask = torch.cat(
                (attention_mask, torch.ones(batch, 1, device=device, dtype=torch.long)), dim=1
            )
            cpost_output = _base_forward(cpost, input_ids, attention_mask, cpost_cache)
            factorized_output = _base_forward(
                factorized, input_ids, attention_mask, factorized_cache
            )
            logits = exact_report(cpost_output.logits, factorized_output.logits)
            cache = _cache_exact(
                cpost_output.past_key_values, factorized_output.past_key_values
            )
            layer_comparisons = []
            first_divergence = None
            if factorized._fpct_packed_layout is None:
                raise R2lGateError("Qwen factorized layout missing during decode")
            for layer in range(layers):
                semantic = factorized._fpct_packed_layout.semantic_parent_equivalent(layer)
                semantic_complete = semantic_complete and semantic is not None and bool(semantic.all())
                cpost_trace = cpost._fpct_attention_trace_tensors[layer][-1]
                factorized_trace = factorized._fpct_attention_trace_tensors[layer][-1]
                for field, left, right in (
                    ("pre_o_proj", cpost_trace["pre_o_proj"], factorized_trace["pre_o_proj"]),
                    ("post_o_proj", cpost_trace["post_o_proj"], factorized_trace["post_o_proj"]),
                    ("residual_hidden", cpost_hidden[layer][-1], factorized_hidden[layer][-1]),
                ):
                    report = exact_report(left, right)
                    row = {"layer": layer, "field": field, **report}
                    layer_comparisons.append(row)
                    if first_divergence is None and not (
                        report["equal"] and report["max_abs"] == 0 and report["ulp_max"] == 0
                    ):
                        first_divergence = row
            step_rows.append(
                {
                    "decode_step": step + 1,
                    "logits": logits,
                    "cache": cache,
                    "layer_endpoint_comparison_count": len(layer_comparisons),
                    "layer_endpoint_records_sha256": hashlib.sha256(
                        json.dumps(layer_comparisons, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "first_divergence": first_divergence,
                }
            )
            cpost_cache = cpost_output.past_key_values
            factorized_cache = factorized_output.past_key_values
    finally:
        for handle in cpost_handles + factorized_handles:
            handle.remove()
    exact = (
        semantic_complete
        and candidate_identity is not None
        and candidate_identity["candidate_kv_parent_bitwise"]
        and all(
            row["logits"]["equal"]
            and row["logits"]["max_abs"] == 0
            and row["logits"]["ulp_max"] == 0
            and row["cache"]["equal"]
            and row["first_divergence"] is None
            for row in step_rows
        )
    )
    return {
        "dtype": str(dtype),
        "layers": layers,
        "kv_heads": kv_heads,
        "decode_steps": decode_steps,
        "prefill_logits": prefill_logits,
        "prefill_cache": prefill_cache,
        "candidate_identity": candidate_identity,
        "semantic_parent_map_complete": semantic_complete,
        "steps": step_rows,
        "exact": exact,
    }


def qwen_decode4(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise R2lGateError("CUDA unavailable")
    output = Path(args.output)
    if output.exists():
        raise R2lGateError("refusing to overwrite Qwen decode4 output")
    rows = [
        run_actual_qwen_decode4_case(device=torch.device("cuda:0"), dtype=dtype)
        for dtype in (torch.float32, torch.bfloat16)
    ]
    payload = {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2l_actual_qwen_decode4_exact_null_v1",
        "status": "GO" if all(row["exact"] for row in rows) else "GPU_ENGINEERING_BLOCKED_R2L",
        "rows": rows,
        "device": torch.cuda.get_device_name(0),
        "accuracy_or_correctness_accessed": False,
    }
    atomic_json(output, payload)
    if payload["status"] != "GO":
        raise R2lGateError("actual Qwen decode4 exact-null check failed")
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


def _immutable_pretrained_semantics(
    conditions_root: Path, floors_path: Path
) -> dict[str, Any]:
    floors = r2.load_json(floors_path)
    tau = {key: float(value["tau"]) for key, value in floors["floors"].items()}
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
        metrics = {
            name: r2._metric_from_trace(forced[2], name)
            for name in ("d_k", "d_v", "candidate_logit_range")
        }
        delta = r2._max_abs(forced[1], forced_rep[1])
        metric_max = {
            name: max((value for _layer, value in rows), default=0.0)
            for name, rows in metrics.items()
        }
        forced_by_dtype[dtype_name] = {
            "delta_fact_max_abs": delta,
            "metrics": metric_max,
            "active": delta > tau["delta_fact_max_abs"]
            and all(metric_max[name] > tau[name] for name in metric_max),
        }
    exact_ok = all(
        row["cpost_vs_f_logits"]["equal"]
        and row["cpost_vs_f_logits"]["max_abs"] == 0
        and row["cpost_vs_f_logits"]["ulp_max"] == 0
        and row["f_vs_replicated_logits"]["equal"]
        and row["f_vs_replicated_logits"]["max_abs"] == 0
        and row["f_vs_replicated_logits"]["ulp_max"] == 0
        and row["cpost_vs_f_layers"]["all_equal"]
        and row["cpost_vs_f_layers"]["first_divergence"] is None
        and row["f_vs_replicated_layers"]["all_equal"]
        and row["f_vs_replicated_layers"]["first_divergence"] is None
        for row in exact_by_dtype.values()
    )
    return {
        "exact_by_dtype": exact_by_dtype,
        "forced_by_dtype": forced_by_dtype,
        "precollapse_candidate_identity": precollapse,
        "exact_ok": exact_ok,
        "active_ok": all(row["active"] for row in forced_by_dtype.values()),
        "precollapse_ok": all(precollapse.values()),
    }


def semantic_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if output.exists():
        raise R2lGateError("refusing to overwrite immutable semantic aggregate")
    original = r2.load_json(Path(args.original_result))
    synthetic_result = r2.load_json(Path(args.synthetic))
    qwen_result = r2.load_json(Path(args.qwen_decode4))
    pretrained = _immutable_pretrained_semantics(
        Path(args.conditions_root), Path(args.floors)
    )
    original_checks = original.get("checks", {})
    original_23 = (
        len(original_checks) == 23
        and all(bool(value) for value in original_checks.values())
        and original.get("status") == "GO"
    )
    checks = {
        "original_23_checks": original_23,
        "native_parent_map_complete": bool(
            synthetic_result["checks"]["native_parent_map_complete"]
        ),
        "unknown_sidecar_fails_closed": bool(
            synthetic_result["checks"]["unknown_sidecar_fails_closed"]
        ),
        "mixed_memory_exact_null_bitwise": bool(
            synthetic_result["checks"]["mixed_memory_exact_null_bitwise"]
        )
        and pretrained["exact_ok"],
        "mixed_batch_exact_active_isolation": bool(
            synthetic_result["checks"]["mixed_batch_exact_active_isolation"]
        ),
        "actual_qwen_decode4_exact_null": qwen_result.get("status") == "GO"
        and all(bool(row.get("exact")) for row in qwen_result.get("rows", [])),
        "active_route_not_bypassed": pretrained["active_ok"],
        "precollapse_candidate_identity": pretrained["precollapse_ok"],
        "original_hot_path_no_sync": bool(original_checks.get("hot_path_no_sync")),
        "original_resource_gate": all(
            bool(original_checks.get(name))
            for name in ("latency_median", "latency_p95", "expansion_mean", "expansion_p95", "hbm")
        ),
    }
    payload = {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2l_immutable_semantic_aggregate_v1",
        "classification": "IMMUTABLE_CONFIRMATORY_GATE",
        "status": "GO" if all(checks.values()) else "GPU_ENGINEERING_BLOCKED_R2L",
        "checks": checks,
        "original_result_status": original.get("status"),
        "original_check_count": len(original_checks),
        "exact_by_dtype": pretrained["exact_by_dtype"],
        "forced_by_dtype": pretrained["forced_by_dtype"],
        "precollapse_candidate_identity": pretrained["precollapse_candidate_identity"],
        "qwen_decode4": qwen_result,
        "accuracy_or_correctness_accessed": False,
        "training_authorized": False,
    }
    atomic_json(output, payload)
    if payload["status"] != "GO":
        raise R2lGateError(f"immutable semantic gate failed: {checks}")
    return payload


def immutable_finalize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if output.exists():
        raise R2lGateError("refusing to overwrite immutable final result")
    original = r2.load_json(Path(args.original_result))
    semantic = r2.load_json(Path(args.semantic_result))
    active = r2.load_json(Path(args.active_aggregate))
    checks = {
        "original_23_checks": original.get("status") == "GO"
        and len(original.get("checks", {})) == 23
        and all(bool(value) for value in original.get("checks", {}).values()),
        "semantic_checks": semantic.get("status") == "GO"
        and all(bool(value) for value in semantic.get("checks", {}).values()),
        "balanced_checkpoint_native": bool(
            active.get("canaries", {}).get("checkpoint_native", {}).get("qualified")
        ),
        "balanced_forced_on": bool(
            active.get("canaries", {}).get("forced_on", {}).get("qualified")
        ),
        "accuracy_firewall": not bool(original.get("accuracy_or_correctness_accessed"))
        and not bool(semantic.get("accuracy_or_correctness_accessed"))
        and not bool(active.get("accuracy_or_correctness_accessed")),
    }
    go = all(checks.values())
    payload = {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2l_immutable_final_v1",
        "classification": "R2L_IMMUTABLE_GO" if go else "GPU_ENGINEERING_BLOCKED_R2L",
        "status": "GO" if go else "GPU_ENGINEERING_BLOCKED_R2L",
        "checks": checks,
        "original_result": original,
        "semantic_result": semantic,
        "balanced_active_canary": active,
        "training_authorized": go,
        "accuracy_or_correctness_accessed": False,
    }
    atomic_json(output, payload)
    if not go:
        raise R2lGateError(f"immutable final gate failed: {checks}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic_parser = commands.add_parser("synthetic")
    synthetic_parser.add_argument("--output", required=True)
    qwen_parser = commands.add_parser("qwen-decode4")
    qwen_parser.add_argument("--output", required=True)
    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--conditions-root", required=True)
    aggregate_parser.add_argument("--floors", required=True)
    aggregate_parser.add_argument("--synthetic", required=True)
    aggregate_parser.add_argument("--latency-block", required=True)
    aggregate_parser.add_argument("--trace-root", required=True)
    aggregate_parser.add_argument("--output", required=True)
    semantic_parser = commands.add_parser("semantic-aggregate")
    semantic_parser.add_argument("--original-result", required=True)
    semantic_parser.add_argument("--conditions-root", required=True)
    semantic_parser.add_argument("--floors", required=True)
    semantic_parser.add_argument("--synthetic", required=True)
    semantic_parser.add_argument("--qwen-decode4", required=True)
    semantic_parser.add_argument("--output", required=True)
    finalize_parser = commands.add_parser("immutable-finalize")
    finalize_parser.add_argument("--original-result", required=True)
    finalize_parser.add_argument("--semantic-result", required=True)
    finalize_parser.add_argument("--active-aggregate", required=True)
    finalize_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "synthetic":
        payload = synthetic(args)
    elif args.command == "qwen-decode4":
        payload = qwen_decode4(args)
    elif args.command == "aggregate":
        payload = aggregate(args)
    elif args.command == "semantic-aggregate":
        payload = semantic_aggregate(args)
    else:
        payload = immutable_finalize(args)
    print(json.dumps({"status": payload["status"], "classification": payload.get("classification")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
