#!/usr/bin/env python3
"""Export outcome-free FPCT-E1 instrumentation hard-gate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

from rosetta.model.fpct_attention import (
    FPCTSidecarSegment,
    fpct_mechanism_diagnostics,
    pack_fpct_memory,
)
from rosetta.model.fpct_instrumentation import (
    FPCTCaptureAccumulator,
    teacher_forced_query_mask,
)
from rosetta.model.wrapper import RosettaModel


SCHEMA_VERSION = 1
SEED = 20260726


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qwen_model(state: dict[str, Tensor] | None = None) -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(1701)
    model = Qwen3ForCausalLM(config)
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return model


def _add_sidecar(wrapper: RosettaModel) -> None:
    generator = torch.Generator().manual_seed(719)
    key = torch.randn(1, 2, 4, 2, 8, generator=generator)
    value = torch.randn(1, 2, 4, 2, 8, generator=generator)
    prior = torch.tensor([[[0.6, 0.4], [1.0, 0.0], [0.7, 0.3], [1.0, 0.0]]])
    wrapper._store_fpct_sidecar(0, 0, key, value, prior, prior > 0)


def _forward(
    wrapper: RosettaModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    cache: DynamicCache | None = None,
    labels: Tensor | None = None,
):
    return wrapper._base_model_forward_with_fpct(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache or DynamicCache(),
        labels=labels,
        use_cache=True,
        return_dict=True,
    )


def _cache_equal(left: DynamicCache, right: DynamicCache) -> bool:
    if left.get_seq_length() != right.get_seq_length():
        return False
    return all(
        torch.equal(a, b)
        for a, b in zip(
            [*left.key_cache, *left.value_cache],
            [*right.key_cache, *right.value_cache],
        )
    )


def parity_evidence() -> dict[str, Any]:
    state = _qwen_model().state_dict()
    baseline = RosettaModel([_qwen_model(state)], fpct_operator="f")
    captured = RosettaModel([_qwen_model(state)], fpct_operator="f")
    _add_sidecar(baseline)
    _add_sidecar(captured)
    ids = torch.tensor([[4, 5, 6, 7]])
    mask = torch.ones(1, 4, dtype=torch.long)
    labels = torch.tensor([[-100, -100, 6, 7]])

    baseline_teacher = _forward(baseline, ids, mask, labels=labels)
    captured.begin_fpct_capture(
        "teacher_forced_response",
        metadata={"gate": "fpct-e1", "sample": "random-qwen3"},
        query_mask=teacher_forced_query_mask(labels),
    )
    captured_teacher = _forward(captured, ids, mask, labels=labels)
    teacher_report = captured.end_fpct_capture()

    logits_equal = torch.equal(baseline_teacher.logits, captured_teacher.logits)
    loss_equal = torch.equal(baseline_teacher.loss, captured_teacher.loss)
    teacher_cache_equal = _cache_equal(
        baseline_teacher.past_key_values, captured_teacher.past_key_values
    )

    baseline_cache: DynamicCache | None = None
    captured_cache: DynamicCache | None = None
    baseline_tokens: list[int] = []
    captured_tokens: list[int] = []
    baseline_input = ids
    captured_input = ids.clone()
    current_mask = mask
    captured.begin_fpct_capture("greedy_decode")
    decode_logits_equal = True
    decode_cache_equal = True
    for step in range(3):
        baseline_output = _forward(
            baseline, baseline_input, current_mask, cache=baseline_cache
        )
        captured_output = _forward(
            captured, captured_input, current_mask, cache=captured_cache
        )
        decode_logits_equal &= torch.equal(
            baseline_output.logits, captured_output.logits
        )
        decode_cache_equal &= _cache_equal(
            baseline_output.past_key_values, captured_output.past_key_values
        )
        baseline_next = int(baseline_output.logits[:, -1].argmax(dim=-1))
        captured_next = int(captured_output.logits[:, -1].argmax(dim=-1))
        baseline_tokens.append(baseline_next)
        captured_tokens.append(captured_next)
        baseline_cache = baseline_output.past_key_values
        captured_cache = captured_output.past_key_values
        baseline_input = torch.tensor([[baseline_next]])
        captured_input = torch.tensor([[captured_next]])
        current_mask = torch.ones(1, ids.shape[1] + step + 1, dtype=torch.long)
    decode_report = captured.end_fpct_capture()

    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "fpct_e1_instrumentation_parity_v1",
        "model": "Qwen3ForCausalLM random tiny config; no pretrained weights",
        "seed": SEED,
        "teacher_forced": {
            "logits_bitwise_equal": logits_equal,
            "loss_bitwise_equal": loss_equal,
            "cache_bitwise_equal": teacher_cache_equal,
            "eligible_query_count": teacher_report["layers"]["0"]["eligible_query_count"],
            "prompt_queries_excluded": True,
        },
        "greedy_decode": {
            "steps": 3,
            "logits_bitwise_equal_each_step": decode_logits_equal,
            "cache_bitwise_equal_each_step": decode_cache_equal,
            "baseline_tokens": baseline_tokens,
            "captured_tokens": captured_tokens,
            "tokens_equal": baseline_tokens == captured_tokens,
            "capture_forward_count": decode_report["forward_count"],
        },
        "capture_contract": {
            "stores_raw_kv": teacher_report["stores_raw_kv"],
            "parent_summaries_present": bool(
                teacher_report["layers"]["0"]["parent_summaries"]
            ),
            "json_serializable": True,
        },
    }
    result["status"] = "GO" if all(
        (
            logits_equal,
            loss_equal,
            teacher_cache_equal,
            decode_logits_equal,
            decode_cache_equal,
            baseline_tokens == captured_tokens,
            not teacher_report["stores_raw_kv"],
            result["capture_contract"]["parent_summaries_present"],
        )
    ) else "BLOCKED"
    return result


def _packed(key: Tensor, value: Tensor, prior: Tensor, query_length: int):
    sidecar = FPCTSidecarSegment(
        parent_start=0,
        key=key,
        value=value,
        prior=prior.float(),
        valid=prior > 0,
    )
    return pack_fpct_memory(
        key[..., 0, :].clone(),
        value[..., 0, :].clone(),
        torch.zeros(1, 1, query_length, key.shape[2]),
        [sidecar],
        query_length=query_length,
    )


def synthetic_evidence() -> dict[str, Any]:
    key = torch.tensor([[[[[2.0, 0.0], [-2.0, 0.0]]]]])
    value = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])
    prior = torch.tensor([[[0.5, 0.5]]])
    packed = _packed(key, value, prior, 1)
    capture = FPCTCaptureAccumulator(
        "teacher_forced_response",
        metadata={"case": "query-changing"},
        query_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    for query in (
        torch.tensor([[[[1.0, 0.0]]]]),
        torch.tensor([[[[-1.0, 0.0]]]]),
    ):
        metrics, payload = fpct_mechanism_diagnostics(
            query, packed, return_capture_payload=True
        )
        capture.update(0, metrics, payload)
    report = capture.finalize()
    parent = report["layers"]["0"]["parent_summaries"][0]

    identical_key = torch.tensor([[[[[1.0, -0.5], [1.0, -0.5]]]]])
    identical_value = torch.tensor([[[[[0.25, 2.0], [0.25, 2.0]]]]])
    identical_prior = torch.tensor([[[0.3, 0.7]]])
    identical_query = torch.tensor([[[[1.0, 0.0], [-1.0, 0.5]]]])
    identical_metrics = fpct_mechanism_diagnostics(
        identical_query,
        _packed(identical_key, identical_value, identical_prior, 2),
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "fpct_e1_synthetic_query_variance_v1",
        "query_changing": {
            "forward_count": report["forward_count"],
            "eligible_query_count": report["layers"]["0"]["eligible_query_count"],
            "gamma_query_variance": report["metrics"]["gamma_query_variance"],
            "posterior_top1_any_change": report["metrics"]["posterior_top1_any_change"],
            "posterior_top1_change_rate": report["metrics"]["posterior_top1_change_rate"],
            "parent_query_count": parent["query_count"],
            "candidate_gamma_moments": parent["candidate_gamma_moments"],
        },
        "identical_candidates": {
            "gamma_kl_prior": float(identical_metrics["gamma_kl_prior"]),
            "gamma_tv_prior": float(identical_metrics["gamma_tv_prior"]),
            "jensen_gap": float(identical_metrics["jensen_gap"]),
        },
        "stores_raw_kv": report["stores_raw_kv"],
    }
    tolerance = 1e-7
    result["status"] = "GO" if (
        result["query_changing"]["gamma_query_variance"] > 0
        and result["query_changing"]["posterior_top1_any_change"]
        and result["query_changing"]["parent_query_count"] == 2
        and abs(result["identical_candidates"]["gamma_kl_prior"]) <= tolerance
        and abs(result["identical_candidates"]["gamma_tv_prior"]) <= tolerance
        and abs(result["identical_candidates"]["jensen_gap"]) <= tolerance
        and not result["stores_raw_kv"]
    ) else "BLOCKED"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-output", type=Path, required=True)
    parser.add_argument("--synthetic-output", type=Path, required=True)
    args = parser.parse_args()
    parity = parity_evidence()
    synthetic = synthetic_evidence()
    atomic_json(args.parity_output, parity)
    atomic_json(args.synthetic_output, synthetic)
    payload = {
        "parity": {"status": parity["status"], "sha256": sha256_file(args.parity_output)},
        "synthetic": {"status": synthetic["status"], "sha256": sha256_file(args.synthetic_output)},
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if parity["status"] == synthetic["status"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
