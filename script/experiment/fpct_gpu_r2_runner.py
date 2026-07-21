from __future__ import annotations

"""Sealed R2 GPU root-cause, numerical and label-free operator runner."""

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable

import torch
from torch.profiler import ProfilerActivity, record_function

from rosetta.model.fpct_attention import (
    FPCTSidecarSegment,
    build_fpct_packed_layout,
    canonical_log_prior,
    fpct_eager_attention,
    fpct_mechanism_diagnostics,
    pack_fpct_memory,
)


CONDITIONS = {
    "OP01_CPOST_NATIVE": {"operator": "c_post"},
    "OP02_F_NATIVE": {"operator": "f", "instrumentation": True},
    "OP03_F_REP_NATIVE": {"operator": "f", "replicated": True, "instrumentation": True},
    "OP04_F_FORCED": {"operator": "f", "forced": True, "instrumentation": True},
    "OP05_F_REP_FORCED": {"operator": "f", "replicated": True, "forced": True, "instrumentation": True},
    "OP06_F_BYPASS": {"operator": "f", "bypass": True},
    "OP07_M1_CPOST": {"operator": "c_post", "m1": True},
    "OP08_M1_F": {"operator": "f", "m1": True},
}
PROFILE_CONDITIONS = {
    "P2_CPOST_OFF": {"operator": "c_post"},
    "P3_F_OFF": {"operator": "f"},
    "P4_F_REPLICATED": {"operator": "f", "replicated": True},
    "P5_F_ON": {"operator": "f", "instrumentation": True},
    "P6_DECODE4": {"operator": "f", "decode": True},
}
TRACE_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


class R2GateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise R2GateError(f"expected JSON object: {path}")
    return value


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_sha256(modules: Iterable[torch.nn.Module]) -> str:
    digest = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, value in sorted(module.state_dict().items()):
            digest.update(f"{module_index}:{name}\0".encode())
            digest.update(bytes.fromhex(tensor_sha256(value)))
    return digest.hexdigest()


def stable_sample_key(task: str, subject: str, question_id: str) -> str:
    payload = {"task": task, "subject": subject, "question_id": question_id}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def content_hash(question: str, choices: list[str]) -> str:
    padded = [normalize_text(choices[i]) if i < min(4, len(choices)) else "" for i in range(10)]
    payload = {"question": normalize_text(question), "choices": padded}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _panel_records(panel: dict[str, Any], data_root: Path) -> list[dict[str, Any]]:
    import pyarrow.ipc as ipc
    import pyarrow.parquet as parquet
    from rosetta.utils.evaluate import build_prompt

    wanted = {row["sample_key_sha256"]: row for row in panel["rows"]}
    found: dict[str, tuple[str, str, list[str]]] = {}

    arc = parquet.read_table(
        data_root / "ai2_arc/ARC-Challenge/test-00000-of-00001.parquet",
        columns=["question", "choices"],
    ).to_pylist()
    for index, raw in enumerate(arc):
        key = stable_sample_key("ai2-arc", "SPLIT_0_OF_1", str(index))
        if key in wanted:
            found[key] = (
                "SPLIT_0_OF_1", str(raw["question"]),
                [str(value) for value in raw["choices"]["text"][:4]],
            )

    obqa = parquet.read_table(
        data_root / "openbookqa/main/test-00000-of-00001.parquet",
        columns=["question_stem", "choices"],
    ).to_pylist()
    for index, raw in enumerate(obqa):
        key = stable_sample_key("openbookqa", "SPLIT_0_OF_1", str(index))
        if key in wanted:
            found[key] = (
                "SPLIT_0_OF_1", str(raw["question_stem"]),
                [str(value) for value in raw["choices"]["text"][:4]],
            )

    for arrow_path in sorted((data_root / "mmlu-redux-2.0").glob("*/data-00000-of-00001.arrow")):
        subject = arrow_path.parent.name
        with arrow_path.open("rb") as handle:
            table = ipc.open_stream(handle).read_all().select(["question", "choices"])
        for index, raw in enumerate(table.to_pylist()):
            key = stable_sample_key("mmlu-redux", subject, str(index))
            if key in wanted:
                found[key] = (
                    subject, str(raw["question"]),
                    [str(value) for value in raw["choices"][:4]],
                )

    records = []
    for row in panel["rows"]:
        key = row["sample_key_sha256"]
        if key not in found:
            raise R2GateError(f"panel sample not found without labels: {key}")
        subject, question, choices = found[key]
        if content_hash(question, choices) != row["content_group_sha256"]:
            raise R2GateError(f"panel content hash mismatch: {row['panel_id']}")
        choice_text = "".join(
            f"{chr(65 + index)}. {choice}\n" for index, choice in enumerate(choices)
        )
        prompt = build_prompt(
            dataset="mmlu-redux", locale="", question=question,
            choices=choice_text, use_cot=False, use_template=True,
        )
        records.append({
            **row,
            "subject": subject,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "The correct answer is"},
            ],
        })
    return records


def _model_config(lock: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any]:
    from script.experiment.fpct_confirmatory_runner import training_config

    config = training_config(
        lock, 104729, condition["operator"], Path("/fpct-run/unused"),
        examples=128, optimizer_steps=4,
    )["model"]
    config.update({
        "attn_implementation": "eager",
        "projector_init_seed": 104729,
        "fpct_replicated_collapse": bool(condition.get("replicated", False)),
        "fpct_collapse_to_parent_bypass": bool(condition.get("bypass", False)),
        "fpct_instrumentation": bool(condition.get("instrumentation", False)),
        "fpct_profile_scopes": True,
        "fpct_trace": True,
    })
    return config


def _assert_eager(model: Any) -> dict[str, Any]:
    roles = {"receiver": model.model_list[0], "sender": model.model_list[1]}
    result = {}
    for role, loaded in roles.items():
        backend = loaded.config._attn_implementation
        if backend != "eager":
            raise R2GateError(f"{role} backend is not eager: {backend!r}")
        layers = [
            type(layer.self_attn).__module__ + "." + type(layer.self_attn).__name__
            for layer in loaded.model.layers
        ]
        result[role] = {
            "backend": backend,
            "attention_classes": layers,
            "layer_count": len(layers),
        }
    return result


def _m1_item(item: dict[str, Any]) -> dict[str, Any]:
    value = item["soft_alignment"]
    indices = value["source_indices"].clone()
    weights = value["source_weights"].clone()
    indices[..., 1:] = -1
    weights[..., 1:] = 0
    weights[..., 0] = torch.where(
        indices[..., 0] >= 0, torch.ones_like(weights[..., 0]),
        torch.zeros_like(weights[..., 0]),
    )
    value["source_indices"] = indices
    value["source_weights"] = weights
    return item


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "input_ids": [value.to(device) for value in batch["input_ids"]],
        "attention_mask": [value.to(device) for value in batch["attention_mask"]],
        "position_ids": batch["position_ids"].to(device),
        "labels": batch["labels"].to(device),
        "kv_cache_index": [value.to(device) for value in batch["kv_cache_index"]],
        "soft_alignment": [
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in section.items()
            }
            for section in batch["soft_alignment"]
        ],
    }


def _forward(model: Any, batch: dict[str, Any]) -> Any:
    return model.forward(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"], labels=None,
        kv_cache_index=batch["kv_cache_index"],
        soft_alignment=batch["soft_alignment"], use_cache=True,
    )


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    value_fp32 = value.detach().float()
    return {
        "sha256": tensor_sha256(value),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "mean": float(value_fp32.mean().cpu()),
        "rms": float(value_fp32.square().mean().sqrt().cpu()),
        "max_abs": float(value_fp32.abs().max().cpu()),
    }


def _trace_summary(model: Any, residuals: dict[int, list[torch.Tensor]]) -> dict[str, Any]:
    result: dict[str, Any] = {"layers": {}}
    layer_ids = sorted(
        set(model._fpct_candidate_trace_tensors)
        | set(model._fpct_attention_trace_tensors)
        | set(residuals)
        | set(model._fpct_layer_metrics)
    )
    for layer in layer_ids:
        row: dict[str, Any] = {}
        candidates = model._fpct_candidate_trace_tensors.get(layer, [])
        if candidates:
            last = candidates[-1]
            for key in (
                "source_candidate_key", "source_candidate_value",
                "native_parent_key", "native_parent_value",
                "fused_candidate_key", "fused_candidate_value",
                "collapsed_key", "collapsed_value", "prior",
                "legacy_key_gate", "legacy_value_gate",
                "parent_force_native",
                "key_alignment_confidence", "value_alignment_confidence",
            ):
                row[key] = _tensor_summary(last[key])
            legal = last["legal"]
            log_prior = torch.where(
                legal, torch.log(last["prior"].float().clamp_min(1e-30)),
                torch.full_like(last["prior"].float(), -torch.inf),
            )
            has = legal.any(dim=-1)
            row["prior_logsumexp_max_abs"] = float(
                torch.where(
                    has, torch.logsumexp(log_prior, -1).abs(),
                    torch.zeros_like(has, dtype=torch.float32),
                ).max().cpu()
            )
            row["prior_sum_max_abs"] = float(
                torch.where(
                    has, (last["prior"].float().sum(-1) - 1).abs(),
                    torch.zeros_like(has, dtype=torch.float32),
                ).max().cpu()
            )
            row["d_k"] = float(
                (last["fused_candidate_key"].float() - last["native_parent_key"].float().unsqueeze(3)).square().mean().sqrt().cpu()
            )
            row["d_v"] = float(
                (last["fused_candidate_value"].float() - last["native_parent_value"].float().unsqueeze(3)).square().mean().sqrt().cpu()
            )
        attention = model._fpct_attention_trace_tensors.get(layer, [])
        if attention:
            row["pre_o_proj"] = _tensor_summary(attention[-1]["pre_o_proj"][:, -1])
            row["post_o_proj"] = _tensor_summary(attention[-1]["post_o_proj"][:, -1])
            if "invalid_probability_max" in attention[-1]:
                row["invalid_probability_max"] = float(
                    attention[-1]["invalid_probability_max"].float().cpu()
                )
        if residuals.get(layer):
            row["residual_hidden"] = _tensor_summary(residuals[layer][-1])
        if layer in model._fpct_layer_metrics:
            row["mechanism"] = {
                key: float(value.detach().float().cpu())
                for key, value in model._fpct_layer_metrics[layer].items()
            }
        result["layers"][str(layer)] = row
    return result


def _compact_trace_tensors(
    model: Any,
    residuals: dict[int, list[torch.Tensor]],
    *,
    parent_index: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Keep only one frozen parent and the last query token per layer."""

    result: dict[str, dict[str, torch.Tensor]] = {}
    layer_ids = sorted(
        set(model._fpct_candidate_trace_tensors)
        | set(model._fpct_attention_trace_tensors)
        | set(residuals)
    )
    for layer in layer_ids:
        row: dict[str, torch.Tensor] = {}
        candidates = model._fpct_candidate_trace_tensors.get(layer, [])
        if candidates:
            last = candidates[-1]
            if parent_index >= last["prior"].shape[1]:
                raise R2GateError(
                    f"trace parent {parent_index} exceeds layer {layer} length"
                )
            for key in (
                "source_candidate_key", "source_candidate_value",
                "fused_candidate_key", "fused_candidate_value",
            ):
                row[key] = last[key][:, :, parent_index].detach().cpu()
            for key in (
                "native_parent_key", "native_parent_value",
                "collapsed_key", "collapsed_value",
            ):
                row[key] = last[key][:, :, parent_index].detach().cpu()
            for key in (
                "prior", "legal", "legacy_key_gate", "legacy_value_gate",
                "parent_force_native",
                "key_alignment_confidence", "value_alignment_confidence",
            ):
                value = last[key]
                if value.ndim >= 3 and value.shape[-2] > parent_index:
                    row[key] = value[..., parent_index, :].detach().cpu()
                else:
                    row[key] = value.detach().cpu()
        attention = model._fpct_attention_trace_tensors.get(layer, [])
        if attention:
            row["pre_o_proj_last"] = attention[-1]["pre_o_proj"][:, -1].detach().cpu()
            row["post_o_proj_last"] = attention[-1]["post_o_proj"][:, -1].detach().cpu()
        if residuals.get(layer):
            row["residual_last"] = residuals[layer][-1].detach().cpu()
        result[str(layer)] = row
    return result


def operator_condition(
    lock_path: Path,
    panel_path: Path,
    data_root: Path,
    condition_id: str,
    dtype_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    if condition_id not in CONDITIONS:
        raise R2GateError(f"unknown condition: {condition_id}")
    if not torch.cuda.is_available():
        raise R2GateError("CUDA unavailable")
    if dtype_name not in TRACE_DTYPES:
        raise R2GateError(f"unknown trace dtype: {dtype_name}")
    condition = CONDITIONS[condition_id]
    model_dtype = TRACE_DTYPES[dtype_name]
    lock = load_json(lock_path)
    panel = load_json(panel_path)
    records = _panel_records(panel, data_root)
    output_dir.mkdir(parents=True, exist_ok=False)

    from script.train import SFT_train as sft

    device = torch.device("cuda:0")
    torch.manual_seed(104729)
    torch.cuda.manual_seed_all(104729)
    model, receiver_tokenizer, aligner, sender_tokenizer = sft.setup_models(
        _model_config(lock, condition), "rosetta", str(device), model_dtype
    )
    model.eval()
    eager = _assert_eager(model)
    projector_hash = state_sha256(model.projector_list)
    gate_logits = [
        {
            "layer": index,
            "key": float(projector.key_gate_logit.detach().float().cpu()),
            "value": float(projector.value_gate_logit.detach().float().cpu()),
        }
        for index, projector in enumerate(model.projector_list)
    ]
    if condition.get("forced"):
        for projector in model.projector_list:
            projector.set_alignment_confidence_eval_mode("forced_on")

    collator = sft.RosettaDataCollator(
        slm_tokenizer=receiver_tokenizer,
        llm_tokenizer=sender_tokenizer,
        max_length=1024,
        aligner=aligner,
        do_alignment=True,
    )
    logits_rows = []
    trace_rows = []
    trace_tensor_rows = []
    expansion_ratios = []
    prior_hashes = set()
    for panel_row in records:
        dataset = sft.AlignedChatDataset(
            [panel_row["messages"]], aligner, max_length=1024,
            soft_alignment_top_k=4,
            fpct_alignment_sanitizer="certified_slot0_v1",
        )
        item = dataset[0]
        observed_m = int(
            (item["soft_alignment"]["source_weights"][panel_row["parent_index"]] > 0).sum()
        )
        if observed_m != panel_row["cardinality"]:
            raise R2GateError(
                f"panel cardinality mismatch {panel_row['panel_id']}: {observed_m}"
            )
        if condition.get("m1"):
            item = _m1_item(item)
        batch = _device_batch(collator([item]), device)
        prior_hashes.update(
            section["fpct_prior_sha256"] for section in batch["soft_alignment"]
        )
        residuals: dict[int, list[torch.Tensor]] = {}
        handles = []
        for layer_index, layer in enumerate(model.model_list[0].model.layers):
            def capture(_module, _inputs, output, *, index=layer_index):
                hidden = output[0] if isinstance(output, tuple) else output
                residuals.setdefault(index, []).append(hidden[:, -1].detach())
            handles.append(layer.register_forward_hook(capture))
        try:
            with torch.no_grad(), record_function("scientific_forward"):
                output = _forward(model, batch)
        finally:
            for handle in handles:
                handle.remove()
        logits_rows.append(output.logits[:, -1].detach().float().cpu())
        trace_rows.append({
            "panel_id": panel_row["panel_id"],
            "prior_sha256": model._fpct_input_prior_sha256,
            "trace": _trace_summary(model, residuals),
        })
        trace_tensor_rows.append({
            "panel_id": panel_row["panel_id"],
            "layers": _compact_trace_tensors(
                model, residuals, parent_index=int(panel_row["parent_index"])
            ),
        })
        layout = model._fpct_packed_layout
        if layout is not None:
            expansion_ratios.extend(
                (
                    layout.expanded_slots.detach().float()
                    / float(layout.source_length)
                ).cpu().tolist()
            )

    logits = torch.cat(logits_rows, dim=0)
    tensor_path = output_dir / "last_token_logits.pt"
    torch.save(logits, tensor_path)
    trace_path = output_dir / "layer_trace.json"
    atomic_json(trace_path, {"rows": trace_rows})
    compact_path = output_dir / "compact_trace_tensors.pt"
    torch.save(trace_tensor_rows, compact_path)
    payload = {
        "schema_version": 1,
        "run_uid": lock["run_uid"],
        "condition_id": condition_id,
        "condition": condition,
        "trace_dtype": dtype_name,
        "status": "COMPLETE",
        "eager_attestation": eager,
        "projector_state_sha256": projector_hash,
        "gate_logits": gate_logits,
        "gate_eval_modes": [
            {
                "alignment": projector.alignment_confidence_eval_mode,
                "legacy": projector.legacy_scalar_gate_eval_mode,
            }
            for projector in model.projector_list
        ],
        "panel_rows_sha256": panel["panel_rows_sha256"],
        "panel_row_count": len(records),
        "canonical_prior_sha256_values": sorted(prior_hashes),
        "logits": {
            "path": str(tensor_path), "sha256": sha256_file(tensor_path),
            "tensor_sha256": tensor_sha256(logits), "shape": list(logits.shape),
        },
        "layer_trace": {
            "path": str(trace_path), "sha256": sha256_file(trace_path),
            "row_count": len(trace_rows),
        },
        "compact_trace_tensors": {
            "path": str(compact_path),
            "sha256": sha256_file(compact_path),
            "row_count": len(trace_tensor_rows),
        },
        "expanded_slot_ratios": expansion_ratios,
        "accuracy_or_correctness_accessed": False,
    }
    atomic_json(output_dir / "condition_manifest.json", payload)
    return payload


def runtime_provenance(lock_path: Path, output: Path) -> dict[str, Any]:
    """Load weights/configs without tokenizers or forward and attest eager."""

    if not torch.cuda.is_available():
        raise R2GateError("CUDA unavailable")
    from transformers import AutoModelForCausalLM
    from rosetta.model.projector import create_projector

    lock = load_json(lock_path)
    device = torch.device("cuda:0")
    receiver_root = lock["assets"]["receiver"]["root"]
    sender_root = lock["assets"]["sender"]["root"]
    receiver = AutoModelForCausalLM.from_pretrained(
        receiver_root, torch_dtype=torch.bfloat16,
        attn_implementation="eager", local_files_only=True,
    ).to(device).eval()
    sender = AutoModelForCausalLM.from_pretrained(
        sender_root, torch_dtype=torch.bfloat16,
        attn_implementation="eager", local_files_only=True,
    ).to(device).eval()
    for role, model in (("receiver", receiver), ("sender", sender)):
        if model.config._attn_implementation != "eager":
            raise R2GateError(f"{role} did not load eager")
    torch.manual_seed(104729)
    prototype = create_projector(
        "C2CProjector",
        source_dim=int(sender.config.head_dim),
        target_dim=int(receiver.config.head_dim),
        source_num_heads=int(sender.config.num_key_value_heads),
        target_num_heads=int(receiver.config.num_key_value_heads),
        hidden_dim=1024, intermediate_dim=1024, num_layers=3, dropout=0.1,
        initial_temperature=1.0, final_temperature=0.001, anneal_steps=64,
        alignment_confidence_gate_mode="token_mlp",
        alignment_confidence_max_delta=2.0,
        dtype=torch.float32,
    )
    layers = int(receiver.config.num_hidden_layers)
    gates = [
        {
            "layer": index,
            "key_gate_logit": float(prototype.key_gate_logit.detach()),
            "value_gate_logit": float(prototype.value_gate_logit.detach()),
            "legacy_scalar_gate_eval_mode": prototype.legacy_scalar_gate_eval_mode,
            "alignment_confidence_eval_mode": prototype.alignment_confidence_eval_mode,
        }
        for index in range(layers)
    ]
    if any(row["key_gate_logit"] != 0 or row["value_gate_logit"] != 0 for row in gates):
        raise R2GateError("fresh projector gate is not exactly zero")
    payload = {
        "schema_version": 1,
        "run_uid": lock["run_uid"],
        "status": "GO",
        "projector_checkpoint_dir": None,
        "projector_state": "fresh",
        "expected_checkpoint_native_classification": "EXPECTED_NATIVE_NULL",
        "gates": gates,
        "receiver": {
            "backend": receiver.config._attn_implementation,
            "attention_classes": [
                type(layer.self_attn).__module__ + "." + type(layer.self_attn).__name__
                for layer in receiver.model.layers
            ],
            "kv_dtype": str(next(receiver.parameters()).dtype),
        },
        "sender": {
            "backend": sender.config._attn_implementation,
            "attention_classes": [
                type(layer.self_attn).__module__ + "." + type(layer.self_attn).__name__
                for layer in sender.model.layers
            ],
            "kv_dtype": str(next(sender.parameters()).dtype),
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "firewall": {
            "tokenizer_loaded": False,
            "natural_prompt_read": False,
            "model_forward": False,
            "accuracy_read": False,
        },
    }
    atomic_json(output, payload)
    return payload


def _plain_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(query.shape[-1])
    logits = logits + mask.float()
    probability = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return torch.matmul(probability, value.float()).to(query.dtype)


def synthetic_null_floors(
    spec_path: Path, receiver_config_path: Path, output: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise R2GateError("CUDA unavailable")
    spec = load_json(spec_path)
    receiver = load_json(receiver_config_path)
    device = torch.device("cuda:0")
    hq = int(receiver["num_attention_heads"])
    hkv = int(receiver["num_key_value_heads"])
    d = int(receiver.get("head_dim") or receiver["hidden_size"] // hq)
    n, q_len, top_k = 8, 4, 4
    values: dict[str, list[float]] = {name: [] for name in spec["metrics"]}
    sample_records = []
    for seed in spec["synthetic_seeds"]:
        for cardinality in spec["cardinalities"]:
            for case in spec["prior_cases"]:
                generator = torch.Generator(device=device).manual_seed(seed)
                for dtype in (torch.float32, torch.bfloat16):
                    query = torch.randn(1, hq, q_len, d, generator=generator, device=device, dtype=dtype)
                    key = torch.randn(1, hkv, n, d, generator=generator, device=device, dtype=dtype)
                    value = torch.randn(1, hkv, n, d, generator=generator, device=device, dtype=dtype)
                    prior = torch.zeros(1, n, top_k, device=device, dtype=torch.float32)
                    if case == "extreme_positive" and cardinality > 1:
                        prior[..., 0] = 1 - 3e-4
                        prior[..., 1:cardinality] = 3e-4 / (cardinality - 1)
                    else:
                        prior[..., :cardinality] = 1.0 / cardinality
                    valid = prior > 0
                    candidate_k = key.unsqueeze(3).expand(1, hkv, n, top_k, d)
                    candidate_v = value.unsqueeze(3).expand_as(candidate_k)
                    sidecar = FPCTSidecarSegment(
                        0, candidate_k, candidate_v, prior, valid,
                        max_slots_hint=n * max(cardinality, 1),
                        source_length_hint=n,
                        prior_sha256="synthetic-null",
                        certified=True,
                    )
                    layout = build_fpct_packed_layout(n, [sidecar])
                    mask = torch.zeros(1, 1, q_len, n, device=device, dtype=torch.float32)
                    if case == "padding":
                        mask[..., -1] = -torch.inf
                    elif case == "causal":
                        mask = torch.triu(
                            torch.full_like(mask, -torch.inf), diagonal=1
                        )
                    packed = pack_fpct_memory(
                        key, value, mask, [sidecar], query_length=q_len, layout=layout
                    )
                    replicated = pack_fpct_memory(
                        key, value, mask, [sidecar], query_length=q_len,
                        layout=layout, replicated_collapse=True,
                    )
                    depth_set = set(spec["depths"])
                    for depth in range(1, max(depth_set) + 1):
                        fact, _ = fpct_eager_attention(query, packed)
                        rep, _ = fpct_eager_attention(query, replicated)
                        bypass = _plain_attention(query, key, value, mask)
                        if depth in depth_set:
                            metrics = fpct_mechanism_diagnostics(query, packed)
                            row = {
                                "seed": seed, "depth": depth,
                                "cardinality": cardinality, "case": case,
                                "dtype": str(dtype),
                                "delta_fact_max_abs": float((fact.float() - rep.float()).abs().max().cpu()),
                                "delta_rep_max_abs": float((rep.float() - bypass.float()).abs().max().cpu()),
                                "delta_bypass_max_abs": 0.0,
                                "gamma_kl_prior": abs(float(metrics["gamma_kl_prior"].cpu())),
                                "jensen_gap": abs(float(metrics["jensen_gap"].cpu())),
                                "gamma_query_variance": abs(float(metrics["gamma_query_variance"].cpu())),
                                "candidate_logit_range": abs(float(metrics["candidate_logit_range"].cpu())),
                                "d_k": abs(float(metrics["d_k"].cpu())),
                                "d_v": abs(float(metrics["d_v"].cpu())),
                            }
                            sample_records.append(row)
                            for name in values:
                                values[name].append(row[name])
                        query = fact
    floors = {}
    for name, metric_spec in spec["metrics"].items():
        tensor = torch.tensor(values[name], dtype=torch.float64)
        quantile = float(torch.quantile(tensor, spec["percentile"] / 100.0))
        floor = max(quantile, float(metric_spec["absolute_floor"])) * float(spec["safety_multiplier"])
        floors[name] = {
            **metric_spec,
            "null_q99_9": quantile,
            "tau": floor,
            "sample_count": len(values[name]),
            "distribution_sha256": hashlib.sha256(
                json.dumps(values[name], separators=(",", ":")).encode()
            ).hexdigest(),
        }
    payload = {
        "schema_version": 1,
        "status": "GO",
        "spec_sha256": sha256_file(spec_path),
        "receiver_config_sha256": sha256_file(receiver_config_path),
        "shape": {"hq": hq, "hkv": hkv, "head_dim": d, "q": q_len, "n": n},
        "floors": floors,
        "all_samples_sha256": hashlib.sha256(
            json.dumps(sample_records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "historical_single_floor_reused": False,
    }
    atomic_json(output, payload)
    return payload


def _profile_sync_events(profile: Any) -> list[dict[str, Any]]:
    result = []
    needles = (
        "aten::item", "_local_scalar_dense", "cudaDeviceSynchronize",
        "cudaStreamSynchronize", "cudaEventSynchronize", "Memcpy DtoH",
    )
    for event in profile.events():
        if not any(needle in event.name for needle in needles):
            continue
        ancestors = []
        parent = event.cpu_parent
        while parent is not None:
            ancestors.append(parent.name)
            parent = parent.cpu_parent
        result.append({
            "name": event.name,
            "cpu_time_us": float(event.cpu_time_total),
            "ancestors": ancestors,
        })
    return result


def _profile_control(positive: bool, trace_path: Path) -> dict[str, Any]:
    value = torch.randn(4096, device="cuda")
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
    ) as profile:
        with record_function("scientific_forward"):
            result = value.square().sum()
            if positive:
                result.item()
    profile.export_chrome_trace(str(trace_path))
    events = _profile_sync_events(profile)
    return {"events": events, "trace_sha256": sha256_file(trace_path)}


def gpu_numerical(
    lock_path: Path, floors_path: Path, output_dir: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise R2GateError("CUDA unavailable")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from rosetta.model.wrapper import RosettaModel
    from script.experiment import fpct_confirmatory_runner as complete_gate

    lock = load_json(lock_path)
    floors = load_json(floors_path)
    if floors.get("status") != "GO":
        raise R2GateError("metric null floors are not GO")
    output_dir.mkdir(parents=True, exist_ok=False)
    complete_path = output_dir / "complete_synthetic_gpu_numerical.json"
    complete_result = complete_gate.gpu_numerical(
        lock_path, complete_path, lock_payload=lock
    )
    device = torch.device("cuda:0")
    results = {}
    for dtype in (torch.float32, torch.bfloat16):
        config = Qwen3Config(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=28, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8,
            max_position_embeddings=64, attention_dropout=0.0, use_cache=True,
        )
        config._attn_implementation = "eager"
        torch.manual_seed(1701)
        base = Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()
        state = base.state_dict()
        cpost = RosettaModel(
            [Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()],
            fpct_operator="c_post",
        )
        cpost.model_list[0].load_state_dict(state)
        replicated = RosettaModel(
            [Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()],
            fpct_operator="f", fpct_replicated_collapse=True,
        )
        replicated.model_list[0].load_state_dict(state)
        bypass = RosettaModel(
            [Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()],
            fpct_operator="f", fpct_collapse_to_parent_bypass=True,
        )
        bypass.model_list[0].load_state_dict(state)
        prior = torch.tensor(
            [[[0.6, 0.4, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
              [0.25, 0.25, 0.25, 0.25], [0.0, 0.0, 0.0, 0.0]]],
            device=device, dtype=torch.float32,
        )
        valid = prior > 0
        for wrapper in (replicated, bypass):
            for layer in range(28):
                candidate = torch.zeros(1, 2, 4, 4, 8, device=device, dtype=dtype)
                wrapper._store_fpct_sidecar(
                    layer, 0, candidate, candidate, prior, valid,
                    prior_sha256="synthetic-qwen-r2",
                    max_slots_hint=8, source_length_hint=4, certified=True,
                )
        ids = torch.tensor([[4, 5, 6, 7]], device=device)
        mask = torch.ones(1, 4, device=device)
        with torch.no_grad():
            cpost_logits = cpost._base_model_forward_with_fpct(
                input_ids=ids, attention_mask=mask, use_cache=True,
                return_dict=True,
            ).logits
            rep_logits = replicated._base_model_forward_with_fpct(
                input_ids=ids, attention_mask=mask, use_cache=True,
                return_dict=True,
            ).logits
            bypass_logits = bypass._base_model_forward_with_fpct(
                input_ids=ids, attention_mask=mask, use_cache=True,
                return_dict=True,
            ).logits
        tolerance = 2e-5 if dtype == torch.float32 else 2e-2
        results[str(dtype)] = {
            "replicated_max_abs": float((rep_logits.float() - cpost_logits.float()).abs().max().cpu()),
            "bypass_max_abs": float((bypass_logits.float() - cpost_logits.float()).abs().max().cpu()),
            "replicated_greedy_equal": bool(torch.equal(rep_logits.argmax(-1), cpost_logits.argmax(-1))),
            "bypass_greedy_equal": bool(torch.equal(bypass_logits.argmax(-1), cpost_logits.argmax(-1))),
            "tolerance": tolerance,
        }
    p0 = _profile_control(False, output_dir / "P0_GPU_NEGATIVE.json.trace")
    p1 = _profile_control(True, output_dir / "P1_ITEM_POSITIVE.json.trace")
    checks = {
        "complete_synthetic_numerical_gate": complete_result["status"] == "GO",
        "actual_qwen_fp32_replicated": results["torch.float32"]["replicated_max_abs"] <= 2e-5,
        "actual_qwen_fp32_bypass": results["torch.float32"]["bypass_max_abs"] <= 2e-5,
        "actual_qwen_bf16_replicated": results["torch.bfloat16"]["replicated_max_abs"] <= 2e-2,
        "actual_qwen_bf16_bypass": results["torch.bfloat16"]["bypass_max_abs"] <= 2e-2,
        "greedy_controls": all(
            value["replicated_greedy_equal"] and value["bypass_greedy_equal"]
            for value in results.values()
        ),
        "profiler_negative_valid": not any(
            "aten::item" in event["name"] or "_local_scalar_dense" in event["name"]
            for event in p0["events"]
        ),
        "profiler_positive_detected": any(
            "aten::item" in event["name"] or "_local_scalar_dense" in event["name"]
            for event in p1["events"]
        ),
    }
    payload = {
        "schema_version": 1,
        "run_uid": lock["run_uid"],
        "status": "GO" if all(checks.values()) else "GPU_ENGINEERING_BLOCKED_R2",
        "checks": checks,
        "actual_qwen_28_layer": results,
        "profile_controls": {"P0_GPU_NEGATIVE": p0, "P1_ITEM_POSITIVE": p1},
        "complete_synthetic_gate": {
            "path": str(complete_path),
            "sha256": sha256_file(complete_path),
            "checks": complete_result["checks"],
            "historical_single_activation_floor_ignored": True,
        },
        "metric_floor_sha256": sha256_file(floors_path),
        "device": torch.cuda.get_device_name(0),
    }
    atomic_json(output_dir / "gpu_numerical.json", payload)
    if payload["status"] != "GO":
        raise R2GateError(f"R2 GPU numerical gate failed: {checks}")
    return payload


def gpu_gate_sequence(
    lock_path: Path,
    floor_spec_path: Path,
    receiver_config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Recoverable pre-pretrained R2 sequence; never reads natural prompts."""

    output_root.mkdir(parents=True, exist_ok=True)
    runtime_path = output_root / "runtime_provenance.json"
    if runtime_path.is_file():
        runtime = load_json(runtime_path)
    else:
        runtime = runtime_provenance(lock_path, runtime_path)
    if runtime.get("status") != "GO":
        raise R2GateError("runtime provenance is not GO")
    gc.collect()
    torch.cuda.empty_cache()

    floors_path = output_root / "metric_null_floors.json"
    if floors_path.is_file():
        floors = load_json(floors_path)
    else:
        floors = synthetic_null_floors(
            floor_spec_path, receiver_config_path, floors_path
        )
    if floors.get("status") != "GO":
        raise R2GateError("metric-specific null floor generation is not GO")
    gc.collect()
    torch.cuda.empty_cache()

    numerical_root = output_root / "gpu_numerical"
    numerical_path = numerical_root / "gpu_numerical.json"
    if numerical_path.is_file():
        numerical = load_json(numerical_path)
    elif numerical_root.exists():
        raise R2GateError("partial GPU numerical directory requires a new run lock")
    else:
        numerical = gpu_numerical(lock_path, floors_path, numerical_root)
    if numerical.get("status") != "GO":
        raise R2GateError("R2 GPU numerical gate is not GO")

    payload = {
        "schema_version": 1,
        "run_uid": load_json(lock_path)["run_uid"],
        "status": "GO",
        "runtime_provenance": {
            "path": str(runtime_path), "sha256": sha256_file(runtime_path),
        },
        "metric_null_floors": {
            "path": str(floors_path), "sha256": sha256_file(floors_path),
        },
        "gpu_numerical": {
            "path": str(numerical_path), "sha256": sha256_file(numerical_path),
        },
        "accuracy_or_correctness_accessed": False,
    }
    atomic_json(output_root / "gpu_gate_sequence.json", payload)
    return payload


def profile_condition(
    lock_path: Path,
    panel_path: Path,
    data_root: Path,
    profile_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    if profile_id not in PROFILE_CONDITIONS:
        raise R2GateError(f"unknown profile: {profile_id}")
    if not torch.cuda.is_available():
        raise R2GateError("CUDA unavailable")
    condition = PROFILE_CONDITIONS[profile_id]
    lock = load_json(lock_path)
    panel = load_json(panel_path)
    panel_row = _panel_records(panel, data_root)[0]
    output_dir.mkdir(parents=True, exist_ok=False)
    from script.train import SFT_train as sft

    device = torch.device("cuda:0")
    config = _model_config(lock, condition)
    config["fpct_trace"] = False
    torch.manual_seed(104729)
    torch.cuda.manual_seed_all(104729)
    model, receiver_tokenizer, aligner, sender_tokenizer = sft.setup_models(
        config, "rosetta", str(device), torch.bfloat16
    )
    model.eval()
    eager = _assert_eager(model)
    collator = sft.RosettaDataCollator(
        slm_tokenizer=receiver_tokenizer,
        llm_tokenizer=sender_tokenizer,
        max_length=1024,
        aligner=aligner,
        do_alignment=True,
    )
    item = sft.AlignedChatDataset(
        [panel_row["messages"]], aligner, max_length=1024,
        soft_alignment_top_k=4,
        fpct_alignment_sanitizer="certified_slot0_v1",
    )[0]
    batch = _device_batch(collator([item]), device)
    with torch.no_grad():
        _forward(model, batch)
    torch.cuda.reset_peak_memory_stats()
    repeats = 7 if profile_id in {"P2_CPOST_OFF", "P3_F_OFF"} else 1
    durations = []
    for _ in range(repeats):
        with record_function("harness_sync"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad(), record_function("scientific_forward"):
            _forward(model, batch)
        with record_function("harness_sync"):
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)

    trace_path = output_dir / f"{profile_id}.chrome.json"
    with record_function("profiler_lifecycle"):
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True,
            record_shapes=True,
        ) as profile:
            with torch.no_grad(), record_function("scientific_forward"):
                if condition.get("decode"):
                    model.generate(
                        kv_cache_index=batch["kv_cache_index"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        position_ids=batch["position_ids"],
                        soft_alignment=batch["soft_alignment"],
                        max_new_tokens=4,
                        temperature=1.0,
                        top_k=1,
                    )
                else:
                    _forward(model, batch)
        profile.export_chrome_trace(str(trace_path))
    events = _profile_sync_events(profile)
    scientific_events = [
        event for event in events if "scientific_forward" in event["ancestors"]
    ]
    hot_events = [
        event for event in events
        if any(
            scope in event["ancestors"]
            for scope in (
                "fpct.pack", "fpct.attention", "fpct.project_candidates"
            )
        )
    ]
    layout = model._fpct_packed_layout
    expansion = None
    if layout is not None:
        expansion = (
            layout.expanded_slots.detach().float()
            / float(layout.source_length)
        ).cpu().tolist()
    payload = {
        "schema_version": 1,
        "run_uid": lock["run_uid"],
        "profile_id": profile_id,
        "condition": condition,
        "status": "COMPLETE",
        "eager_attestation": eager,
        "latency_samples_seconds": durations,
        "latency_median_seconds": statistics.median(durations),
        "latency_p95_seconds": sorted(durations)[-1],
        "peak_hbm_gib": torch.cuda.max_memory_allocated() / 2**30,
        "expanded_slot_ratios": expansion,
        "events": events,
        "scientific_sync_event_count": len(scientific_events),
        "hot_path_sync_event_count": len(hot_events),
        "trace": {
            "path": str(trace_path),
            "sha256": sha256_file(trace_path),
            "bytes": trace_path.stat().st_size,
        },
        "accuracy_or_correctness_accessed": False,
    }
    atomic_json(output_dir / "profile_manifest.json", payload)
    return payload


def _load_condition(
    root: Path, dtype_name: str, condition_id: str,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any], list[dict[str, Any]]]:
    directory = root / "conditions" / dtype_name / condition_id
    manifest = load_json(directory / "condition_manifest.json")
    logits = torch.load(directory / "last_token_logits.pt", map_location="cpu", weights_only=True)
    trace = load_json(directory / "layer_trace.json")
    compact = torch.load(
        directory / "compact_trace_tensors.pt",
        map_location="cpu",
        weights_only=True,
    )
    return manifest, logits, trace, compact


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def _metric_from_trace(trace: dict[str, Any], metric: str, reduction: str = "max") -> list[tuple[int, float]]:
    values = []
    for sample in trace["rows"]:
        for layer_text, layer in sample["trace"]["layers"].items():
            layer_index = int(layer_text)
            if metric in {"d_k", "d_v"} and metric in layer:
                values.append((layer_index, float(layer[metric])))
                continue
            mechanism = layer.get("mechanism", {})
            key = f"{metric}/{reduction}"
            if key in mechanism:
                values.append((layer_index, float(mechanism[key])))
    return values


def _compact_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        panel_id = str(row["panel_id"])
        if panel_id in result:
            raise R2GateError(f"duplicate compact trace row: {panel_id}")
        result[panel_id] = row["layers"]
    return result


def _precollapse_identity(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]],
) -> bool:
    left = _compact_index(left_rows)
    right = _compact_index(right_rows)
    if set(left) != set(right):
        return False
    for panel_id in sorted(left):
        if set(left[panel_id]) != set(right[panel_id]):
            return False
        for layer in sorted(left[panel_id], key=int):
            for key in ("fused_candidate_key", "fused_candidate_value"):
                if key not in left[panel_id][layer] or key not in right[panel_id][layer]:
                    return False
                if not torch.equal(
                    left[panel_id][layer][key], right[panel_id][layer][key]
                ):
                    return False
    return True


def _layer_delta_report(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    left = _compact_index(left_rows)
    right = _compact_index(right_rows)
    if set(left) != set(right):
        raise R2GateError("compact trace panel mismatch")
    fields = ("pre_o_proj_last", "post_o_proj_last", "residual_last")
    maximum = {field: 0.0 for field in fields}
    first: dict[str, dict[str, Any] | None] = {field: None for field in fields}
    records = []
    for panel_id in sorted(left):
        if set(left[panel_id]) != set(right[panel_id]):
            raise R2GateError(f"compact trace layer mismatch: {panel_id}")
        for layer_text in sorted(left[panel_id], key=int):
            layer = int(layer_text)
            row = {"panel_id": panel_id, "layer": layer}
            for field in fields:
                if field not in left[panel_id][layer_text] or field not in right[panel_id][layer_text]:
                    continue
                delta = float(
                    (
                        left[panel_id][layer_text][field].float()
                        - right[panel_id][layer_text][field].float()
                    ).abs().max()
                )
                row[field] = delta
                maximum[field] = max(maximum[field], delta)
                if delta > tolerance and (
                    first[field] is None or layer < int(first[field]["layer"])
                ):
                    first[field] = {
                        "panel_id": panel_id,
                        "layer": layer,
                        "max_abs": delta,
                    }
            records.append(row)
    return {
        "tolerance": tolerance,
        "max_abs": maximum,
        "first_above_tolerance": first,
        "records_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def aggregate_pretrained(
    lock_path: Path, floors_path: Path, gpu_gate_path: Path,
    root: Path, output: Path,
) -> dict[str, Any]:
    lock = load_json(lock_path)
    floors = load_json(floors_path)
    gpu = load_json(gpu_gate_path)
    if gpu.get("status") != "GO":
        raise R2GateError("pretrained aggregate requires GPU numerical GO")
    conditions = {
        dtype_name: {
            condition_id: _load_condition(root, dtype_name, condition_id)
            for condition_id in CONDITIONS
        }
        for dtype_name in TRACE_DTYPES
    }
    profiles = {
        profile_id: load_json(root / "profiles" / profile_id / "profile_manifest.json")
        for profile_id in PROFILE_CONDITIONS
    }
    manifests = {
        dtype_name: {key: value[0] for key, value in rows.items()}
        for dtype_name, rows in conditions.items()
    }
    logits = {
        dtype_name: {key: value[1] for key, value in rows.items()}
        for dtype_name, rows in conditions.items()
    }
    traces = {
        dtype_name: {key: value[2] for key, value in rows.items()}
        for dtype_name, rows in conditions.items()
    }
    compact = {
        dtype_name: {key: value[3] for key, value in rows.items()}
        for dtype_name, rows in conditions.items()
    }
    projector_hashes = {
        dtype_name: {
            value["projector_state_sha256"] for value in manifests[dtype_name].values()
        }
        for dtype_name in TRACE_DTYPES
    }
    panel_hashes = {
        value["panel_rows_sha256"]
        for dtype_rows in manifests.values()
        for value in dtype_rows.values()
    }
    eager_ok = all(
        role["backend"] == "eager"
        for dtype_rows in manifests.values()
        for manifest in dtype_rows.values()
        for role in manifest["eager_attestation"].values()
    )
    prior_invariants = []
    invalid_probability_maxima = []
    for dtype_rows in traces.values():
        for trace in dtype_rows.values():
            for sample in trace["rows"]:
                for layer in sample["trace"]["layers"].values():
                    if "prior_logsumexp_max_abs" in layer:
                        prior_invariants.extend([
                            float(layer["prior_logsumexp_max_abs"]),
                            float(layer["prior_sum_max_abs"]),
                        ])
                    if "invalid_probability_max" in layer:
                        invalid_probability_maxima.append(
                            float(layer["invalid_probability_max"])
                        )
    tau = {name: value["tau"] for name, value in floors["floors"].items()}
    deltas = {}
    forced_metrics = {}
    layer_deltas = {}
    precollapse = {}
    replicated_local_numeric = {}
    for dtype_name in TRACE_DTYPES:
        dtype_logits = logits[dtype_name]
        dtype_traces = traces[dtype_name]
        deltas[dtype_name] = {
            "delta_fact_checkpoint_native_max_abs": _max_abs(
                dtype_logits["OP02_F_NATIVE"], dtype_logits["OP03_F_REP_NATIVE"]
            ),
            "delta_fact_forced_max_abs": _max_abs(
                dtype_logits["OP04_F_FORCED"], dtype_logits["OP05_F_REP_FORCED"]
            ),
            "delta_rep_max_abs": _max_abs(
                dtype_logits["OP03_F_REP_NATIVE"], dtype_logits["OP01_CPOST_NATIVE"]
            ),
            "delta_bypass_max_abs": _max_abs(
                dtype_logits["OP06_F_BYPASS"], dtype_logits["OP01_CPOST_NATIVE"]
            ),
            "m1_max_abs": _max_abs(
                dtype_logits["OP08_M1_F"], dtype_logits["OP07_M1_CPOST"]
            ),
        }
        forced_metrics[dtype_name] = {
            "d_k": _metric_from_trace(dtype_traces["OP04_F_FORCED"], "d_k"),
            "d_v": _metric_from_trace(dtype_traces["OP04_F_FORCED"], "d_v"),
            "candidate_logit_range": _metric_from_trace(
                dtype_traces["OP04_F_FORCED"], "candidate_logit_range"
            ),
            "posterior": (
                _metric_from_trace(dtype_traces["OP04_F_FORCED"], "gamma_kl_prior")
                + _metric_from_trace(dtype_traces["OP04_F_FORCED"], "gamma_query_variance")
                + _metric_from_trace(dtype_traces["OP04_F_FORCED"], "jensen_gap")
            ),
        }
        tolerance = 2e-5 if dtype_name == "fp32" else 2e-2
        layer_deltas[dtype_name] = {
            "replicated_vs_cpost": _layer_delta_report(
                compact[dtype_name]["OP03_F_REP_NATIVE"],
                compact[dtype_name]["OP01_CPOST_NATIVE"],
                tolerance=tolerance,
            ),
            "bypass_vs_cpost": _layer_delta_report(
                compact[dtype_name]["OP06_F_BYPASS"],
                compact[dtype_name]["OP01_CPOST_NATIVE"],
                tolerance=tolerance,
            ),
            "m1_f_vs_cpost": _layer_delta_report(
                compact[dtype_name]["OP08_M1_F"],
                compact[dtype_name]["OP07_M1_CPOST"],
                tolerance=tolerance,
            ),
            "real_vs_replicated_forced": _layer_delta_report(
                compact[dtype_name]["OP04_F_FORCED"],
                compact[dtype_name]["OP05_F_REP_FORCED"],
                tolerance=tolerance,
            ),
        }
        precollapse[dtype_name] = _precollapse_identity(
            compact[dtype_name]["OP01_CPOST_NATIVE"],
            compact[dtype_name]["OP02_F_NATIVE"],
        ) and _precollapse_identity(
            compact[dtype_name]["OP04_F_FORCED"],
            compact[dtype_name]["OP05_F_REP_FORCED"],
        )
        replicated_local_numeric[dtype_name] = _metric_from_trace(
            dtype_traces["OP03_F_REP_NATIVE"],
            "replicated_expanded_delta",
        )
    nonfinal = lambda rows, threshold: any(
        layer < 27 and value > threshold for layer, value in rows
    )

    p2, p3, p4 = (
        profiles["P2_CPOST_OFF"], profiles["P3_F_OFF"],
        profiles["P4_F_REPLICATED"],
    )
    latency_median_ratio = p3["latency_median_seconds"] / p2["latency_median_seconds"]
    latency_p95_ratio = p3["latency_p95_seconds"] / p2["latency_p95_seconds"]
    scientific_sync_baseline = p2["scientific_sync_event_count"]
    geometry = lock.get("resource_geometry", {}).get("tinyllama_all_splits", {})
    geometry_rows = list(geometry.get("tasks", {}).values())
    checks = {
        "eager_runtime": eager_ok,
        "projector_state_identity": all(
            len(projector_hashes[dtype_name]) == 1 for dtype_name in TRACE_DTYPES
        ),
        "panel_identity": len(panel_hashes) == 1,
        "canonical_prior": bool(prior_invariants) and max(prior_invariants) <= 2e-7,
        "finite": all(
            bool(torch.isfinite(value).all())
            for dtype_rows in logits.values()
            for value in dtype_rows.values()
        ),
        "no_mask_leakage": bool(invalid_probability_maxima)
        and max(invalid_probability_maxima) == 0.0,
        "expected_native_null": all(
            deltas[dtype_name]["delta_fact_checkpoint_native_max_abs"]
            <= tau["delta_fact_max_abs"]
            and all(
                row["key"] == 0 and row["value"] == 0
                for row in manifests[dtype_name]["OP02_F_NATIVE"]["gate_logits"]
            )
            for dtype_name in TRACE_DTYPES
        ),
        "forced_on_dk": all(
            nonfinal(forced_metrics[name]["d_k"], tau["d_k"])
            for name in TRACE_DTYPES
        ),
        "forced_on_dv": all(
            nonfinal(forced_metrics[name]["d_v"], tau["d_v"])
            for name in TRACE_DTYPES
        ),
        "forced_on_logit_range": all(
            nonfinal(
                forced_metrics[name]["candidate_logit_range"],
                tau["candidate_logit_range"],
            ) for name in TRACE_DTYPES
        ),
        "forced_on_query_activation": all(
            deltas[name]["delta_fact_forced_max_abs"] > tau["delta_fact_max_abs"]
            or nonfinal(
                forced_metrics[name]["posterior"],
                min(
                    tau["gamma_kl_prior"], tau["gamma_query_variance"],
                    tau["jensen_gap"],
                ),
            ) for name in TRACE_DTYPES
        ),
        "precollapse_candidate_identity": all(precollapse.values()),
        "collapse_bypass": all(
            deltas[name]["delta_bypass_max_abs"] <= tau["delta_bypass_max_abs"]
            and all(
                value is None
                for value in layer_deltas[name]["bypass_vs_cpost"]["first_above_tolerance"].values()
            ) for name in TRACE_DTYPES
        ),
        "replicated_atoms": all(
            deltas[name]["delta_rep_max_abs"] <= tau["delta_rep_max_abs"]
            and all(
                value is None
                for value in layer_deltas[name]["replicated_vs_cpost"]["first_above_tolerance"].values()
            ) for name in TRACE_DTYPES
        ),
        "replicated_local_numeric": all(
            bool(replicated_local_numeric[name])
            and max(value for _layer, value in replicated_local_numeric[name])
            <= (2e-5 if name == "fp32" else 2e-2)
            for name in TRACE_DTYPES
        ),
        "m1_control": all(
            deltas[name]["m1_max_abs"] <= tau["delta_rep_max_abs"]
            and all(
                value is None
                for value in layer_deltas[name]["m1_f_vs_cpost"]["first_above_tolerance"].values()
            ) for name in TRACE_DTYPES
        ),
        "hot_path_no_sync": all(
            profile["hot_path_sync_event_count"] == 0 for profile in profiles.values()
        ),
        "f_no_new_scientific_sync": (
            p3["scientific_sync_event_count"] <= scientific_sync_baseline
            and p4["scientific_sync_event_count"] <= scientific_sync_baseline
        ),
        "latency_median": latency_median_ratio <= 1.50,
        "latency_p95": latency_p95_ratio <= 1.75,
        "hbm": max(profile["peak_hbm_gib"] for profile in profiles.values())
        < 0.9 * torch.cuda.get_device_properties(0).total_memory / 2**30,
        "expansion_mean": bool(geometry_rows) and all(
            float(row["mean"]) <= 1.35 for row in geometry_rows
        ),
        "expansion_p95": bool(geometry_rows) and all(
            float(row["p95"]) <= 1.50 for row in geometry_rows
        ),
    }
    payload = {
        "schema_version": 1,
        "run_uid": lock["run_uid"],
        "status": "GO" if all(checks.values()) else "GPU_ENGINEERING_BLOCKED_R2",
        "checks": checks,
        "deltas": deltas,
        "layer_first_divergence": layer_deltas,
        "precollapse_candidate_identity_by_dtype": precollapse,
        "replicated_local_numeric_by_dtype": {
            name: {
                "tolerance": 2e-5 if name == "fp32" else 2e-2,
                "maximum": max(
                    (value for _layer, value in rows), default=None
                ),
                "layer_count": len(rows),
            }
            for name, rows in replicated_local_numeric.items()
        },
        "resource": {
            "latency_median_ratio": latency_median_ratio,
            "latency_p95_ratio": latency_p95_ratio,
            "peak_hbm_gib": max(profile["peak_hbm_gib"] for profile in profiles.values()),
            "certified_geometry": geometry,
        },
        "projector_state_sha256": {
            name: next(iter(values)) if len(values) == 1 else None
            for name, values in projector_hashes.items()
        },
        "metric_floor_sha256": sha256_file(floors_path),
        "accuracy_or_correctness_accessed": False,
    }
    atomic_json(output, payload)
    if payload["status"] != "GO":
        raise R2GateError(f"R2 pretrained gate failed: {checks}")
    return payload


def _sealed_subprocess(arguments: list[str]) -> None:
    repo = Path(__file__).resolve().parents[2]
    python = Path(sys.executable).resolve()
    bootstrap = repo / "script/runtime/fpct_bootstrap.py"
    target = Path(__file__).resolve()
    command = [
        str(python), "-I", str(bootstrap), "--repo-root", str(repo),
        "--target", str(target), "--include-gpu-closure", "--", *arguments,
    ]
    subprocess.run(command, cwd=repo, env=dict(os.environ), check=True)


def pretrained_matrix(
    lock_path: Path, panel_path: Path, data_root: Path,
    floors_path: Path, gpu_gate_path: Path, runtime_path: Path,
    root: Path,
) -> dict[str, Any]:
    runtime = load_json(runtime_path)
    gpu = load_json(gpu_gate_path)
    if runtime.get("status") != "GO" or gpu.get("status") != "GO":
        raise R2GateError("pretrained matrix prerequisites are not GO")
    root.mkdir(parents=True, exist_ok=True)
    for dtype_name in TRACE_DTYPES:
        for condition_id in CONDITIONS:
            directory = root / "conditions" / dtype_name / condition_id
            manifest = directory / "condition_manifest.json"
            if manifest.is_file():
                existing = load_json(manifest)
                if (
                    existing.get("condition_id") != condition_id
                    or existing.get("trace_dtype") != dtype_name
                ):
                    raise R2GateError(f"condition manifest mismatch: {manifest}")
                continue
            _sealed_subprocess([
                "operator-condition", "--run-lock", str(lock_path),
                "--panel", str(panel_path), "--data-root", str(data_root),
                "--condition", condition_id, "--dtype", dtype_name,
                "--output-dir", str(directory),
            ])
    for profile_id in PROFILE_CONDITIONS:
        directory = root / "profiles" / profile_id
        manifest = directory / "profile_manifest.json"
        if manifest.is_file():
            existing = load_json(manifest)
            if existing.get("profile_id") != profile_id:
                raise R2GateError(f"profile manifest mismatch: {manifest}")
            continue
        _sealed_subprocess([
            "profile-condition", "--run-lock", str(lock_path),
            "--panel", str(panel_path), "--data-root", str(data_root),
            "--profile", profile_id, "--output-dir", str(directory),
        ])
    result = aggregate_pretrained(
        lock_path, floors_path, gpu_gate_path, root,
        root / "pretrained_r2_result.json",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    runtime = sub.add_parser("runtime-provenance")
    runtime.add_argument("--run-lock", type=Path, required=True)
    runtime.add_argument("--output", type=Path, required=True)
    floors = sub.add_parser("null-floors")
    floors.add_argument("--spec", type=Path, required=True)
    floors.add_argument("--receiver-config", type=Path, required=True)
    floors.add_argument("--output", type=Path, required=True)
    gpu = sub.add_parser("gpu-numerical")
    gpu.add_argument("--run-lock", type=Path, required=True)
    gpu.add_argument("--floors", type=Path, required=True)
    gpu.add_argument("--output-dir", type=Path, required=True)
    sequence = sub.add_parser("gpu-gate-sequence")
    sequence.add_argument("--run-lock", type=Path, required=True)
    sequence.add_argument("--floor-spec", type=Path, required=True)
    sequence.add_argument("--receiver-config", type=Path, required=True)
    sequence.add_argument("--output-root", type=Path, required=True)
    condition = sub.add_parser("operator-condition")
    condition.add_argument("--run-lock", type=Path, required=True)
    condition.add_argument("--panel", type=Path, required=True)
    condition.add_argument("--data-root", type=Path, required=True)
    condition.add_argument("--condition", required=True)
    condition.add_argument("--dtype", choices=tuple(TRACE_DTYPES), required=True)
    condition.add_argument("--output-dir", type=Path, required=True)
    profile = sub.add_parser("profile-condition")
    profile.add_argument("--run-lock", type=Path, required=True)
    profile.add_argument("--panel", type=Path, required=True)
    profile.add_argument("--data-root", type=Path, required=True)
    profile.add_argument("--profile", required=True)
    profile.add_argument("--output-dir", type=Path, required=True)
    aggregate = sub.add_parser("aggregate-pretrained")
    aggregate.add_argument("--run-lock", type=Path, required=True)
    aggregate.add_argument("--floors", type=Path, required=True)
    aggregate.add_argument("--gpu-gate", type=Path, required=True)
    aggregate.add_argument("--root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    matrix = sub.add_parser("pretrained-matrix")
    matrix.add_argument("--run-lock", type=Path, required=True)
    matrix.add_argument("--panel", type=Path, required=True)
    matrix.add_argument("--data-root", type=Path, required=True)
    matrix.add_argument("--floors", type=Path, required=True)
    matrix.add_argument("--gpu-gate", type=Path, required=True)
    matrix.add_argument("--runtime-provenance", type=Path, required=True)
    matrix.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "runtime-provenance":
        payload = runtime_provenance(args.run_lock, args.output)
    elif args.command == "null-floors":
        payload = synthetic_null_floors(args.spec, args.receiver_config, args.output)
    elif args.command == "gpu-numerical":
        payload = gpu_numerical(args.run_lock, args.floors, args.output_dir)
    elif args.command == "gpu-gate-sequence":
        payload = gpu_gate_sequence(
            args.run_lock, args.floor_spec, args.receiver_config,
            args.output_root,
        )
    elif args.command == "operator-condition":
        payload = operator_condition(
            args.run_lock, args.panel, args.data_root,
            args.condition, args.dtype, args.output_dir,
        )
    elif args.command == "profile-condition":
        payload = profile_condition(
            args.run_lock, args.panel, args.data_root,
            args.profile, args.output_dir,
        )
    elif args.command == "aggregate-pretrained":
        payload = aggregate_pretrained(
            args.run_lock, args.floors, args.gpu_gate, args.root, args.output,
        )
    else:
        payload = pretrained_matrix(
            args.run_lock, args.panel, args.data_root, args.floors,
            args.gpu_gate, args.runtime_provenance, args.root,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
