from __future__ import annotations

"""Sealed target for FPCT GPU gates and matched seed triplets."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import statistics
from typing import Any

import torch

try:
    from fpct_bootstrap import loaded_module, require_active
except ModuleNotFoundError:  # import-only unit tests; execution still fails require_active
    from script.runtime.fpct_bootstrap import loaded_module, require_active
from rosetta.model.fpct_attention import (
    FPCTSidecarSegment,
    build_fpct_packed_layout,
    fpct_eager_attention,
    pack_fpct_memory,
)
from script.analysis import fpct_reference_operator as reference_operator


ARM_ORDER = {
    45: ("c_pre", "c_post", "f"), 46: ("c_pre", "f", "c_post"),
    47: ("c_post", "c_pre", "f"), 48: ("c_post", "f", "c_pre"),
    49: ("f", "c_pre", "c_post"), 50: ("f", "c_post", "c_pre"),
    51: ("c_pre", "c_post", "f"), 52: ("c_pre", "f", "c_post"),
    53: ("c_post", "c_pre", "f"), 54: ("c_post", "f", "c_pre"),
    55: ("f", "c_pre", "c_post"), 56: ("f", "c_post", "c_pre"),
}


class GateError(RuntimeError):
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_lock(path: Path) -> dict[str, Any]:
    require_active(target=Path(__file__))
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"run_uid", "scientific_code_commit", "image", "manifest_sha256"}
    if not required.issubset(payload):
        raise GateError("incomplete confirmatory run lock")
    return payload


def make_case(device: torch.device, dtype: torch.dtype):
    generator = torch.Generator(device=device).manual_seed(20260719)
    q = torch.randn(2, 4, 3, 16, generator=generator, device=device, dtype=dtype, requires_grad=True)
    native_k = torch.randn(2, 2, 5, 16, generator=generator, device=device, dtype=dtype, requires_grad=True)
    native_v = torch.randn(2, 2, 5, 16, generator=generator, device=device, dtype=dtype, requires_grad=True)
    fused_k = torch.randn(2, 2, 5, 4, 16, generator=generator, device=device, dtype=dtype, requires_grad=True)
    fused_v = torch.randn(2, 2, 5, 4, 16, generator=generator, device=device, dtype=dtype, requires_grad=True)
    prior = torch.tensor([
        [[.55,.45,0,0],[1,0,0,0],[.2,.3,.5,0],[0,0,0,0],[.25,.25,.25,.25]],
        [[1,0,0,0],[.7,.3,0,0],[0,0,0,0],[.4,.6,0,0],[1,0,0,0]],
    ], device=device, dtype=dtype)
    valid = prior > 0
    collapsed_k = (fused_k * prior[:, None, :, :, None]).sum(3)
    collapsed_v = (fused_v * prior[:, None, :, :, None]).sum(3)
    supported = valid.any(-1)[:, None, :, None]
    cache_k = torch.where(supported, collapsed_k, native_k)
    cache_v = torch.where(supported, collapsed_v, native_v)
    mask = torch.zeros(2, 1, 3, 5, device=device, dtype=dtype)
    mask[:, :, 0, 3:] = -torch.inf
    mask[1, :, :, 0] = -torch.inf
    sidecar = FPCTSidecarSegment(0, fused_k, fused_v, prior, valid)
    return q, native_k, native_v, cache_k, cache_v, fused_k, fused_v, prior, valid, mask, sidecar


def _identity_fuser(_base_k, _base_v, source_k, source_v):
    return source_k, source_v


def _plain_attention(q: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    hq, hkv = q.shape[1], key.shape[1]
    expanded_key = key.repeat_interleave(hq // hkv, dim=1)
    expanded_value = value.repeat_interleave(hq // hkv, dim=1)
    accumulation = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    logits = torch.einsum("bhqd,bhmd->bhqm", q.to(accumulation), expanded_key.to(accumulation)) / (q.shape[-1] ** .5)
    logits = logits + mask.to(accumulation)
    probability = torch.softmax(logits, dim=-1)
    return torch.einsum("bhqm,bhmd->bhqd", probability, expanded_value.to(accumulation)).to(q.dtype)


def run_one(dtype: torch.dtype) -> dict[str, Any]:
    device = torch.device("cuda")
    q, native_k, native_v, key, value, fused_k, fused_v, prior, valid, mask, sidecar = make_case(device, dtype)
    layout = build_fpct_packed_layout(key.shape[2], [sidecar])
    packed = pack_fpct_memory(key, value, mask, [sidecar], query_length=q.shape[2], layout=layout)
    output, probability = fpct_eager_attention(q, packed)
    loss = output.float().square().sum()
    gradients = torch.autograd.grad(loss, (q, sidecar.key, sidecar.value), retain_graph=False)
    invalid = ~sidecar.valid
    invalid_k = gradients[1].masked_select(invalid[:, None, :, :, None].expand_as(gradients[1]))
    invalid_v = gradients[2].masked_select(invalid[:, None, :, :, None].expand_as(gradients[2]))
    replicated = pack_fpct_memory(key, value, mask, [sidecar], query_length=q.shape[2], layout=layout, replicated_collapse=True)
    replicated_output, _ = fpct_eager_attention(q.detach(), replicated)
    cpost_output = _plain_attention(q.detach(), key.detach(), value.detach(), mask)
    replicated_delta = float((replicated_output.float() - cpost_output.float()).abs().max().cpu())

    first = valid.to(torch.long).argmax(dim=-1)
    has = valid.any(dim=-1)
    onehot = torch.zeros_like(prior).scatter(-1, first.unsqueeze(-1), 1.0)
    onehot = torch.where(has.unsqueeze(-1), onehot, torch.zeros_like(onehot))
    onehot_valid = onehot > 0
    gather_index = first[:, None, :, None, None].expand(fused_k.shape[0], fused_k.shape[1], fused_k.shape[2], 1, fused_k.shape[-1])
    first_k = torch.gather(fused_k.detach(), 3, gather_index).squeeze(3)
    first_v = torch.gather(fused_v.detach(), 3, gather_index).squeeze(3)
    onehot_cache_k = torch.where(has[:, None, :, None], first_k, native_k.detach())
    onehot_cache_v = torch.where(has[:, None, :, None], first_v, native_v.detach())
    onehot_sidecar = FPCTSidecarSegment(0, fused_k.detach(), fused_v.detach(), onehot, onehot_valid)
    onehot_layout = build_fpct_packed_layout(onehot_cache_k.shape[2], [onehot_sidecar])
    onehot_packed = pack_fpct_memory(onehot_cache_k, onehot_cache_v, mask, [onehot_sidecar], query_length=q.shape[2], layout=onehot_layout)
    onehot_f, _ = fpct_eager_attention(q.detach(), onehot_packed)
    onehot_post = _plain_attention(q.detach(), onehot_cache_k, onehot_cache_v, mask)
    onehot_delta = float((onehot_f.float() - onehot_post.float()).abs().max().cpu())

    reference_output = None
    reference_gradient_relative_l2 = None
    if dtype == torch.float32:
        reference_result = reference_operator.f_flat(
            q, native_k, native_v, fused_k, fused_v, prior, valid,
            _identity_fuser, None, mask,
        )
        reference_output = reference_result.output
        reference_gradients = torch.autograd.grad(
            reference_output.float().square().sum(), (q, fused_k, fused_v)
        )
        reference_gradient_relative_l2 = max(
            relative_l2(actual.detach().float().cpu(), expected.detach().float().cpu())
            for actual, expected in zip(gradients, reference_gradients)
        )
        reference_output = reference_output.detach().float().cpu()
    return {
        "output": output.detach().float().cpu(),
        "probability": probability.detach().float().cpu(),
        "gradients": [item.detach().float().cpu() for item in gradients],
        "row_sum_error": float((probability.float().sum(-1) - 1).abs().max().detach().cpu()),
        "invalid_gradient_exact_zero": bool(torch.count_nonzero(invalid_k) == 0 and torch.count_nonzero(invalid_v) == 0),
        "finite": bool(torch.isfinite(output).all() and torch.isfinite(probability).all()),
        "expanded_slots": layout.expanded_slots.detach().cpu().tolist(),
        "replicated_collapse_output_delta": replicated_delta,
        "replicated_collapse_greedy_equal": bool(torch.equal(replicated_output.argmax(-1), cpost_output.argmax(-1))),
        "m1_output_delta": onehot_delta,
        "m1_greedy_equal": bool(torch.equal(onehot_f.argmax(-1), onehot_post.argmax(-1))),
        "reference_output": reference_output,
        "reference_gradient_relative_l2": reference_gradient_relative_l2,
    }


def relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-30))


def gpu_numerical(lock_path: Path, output_path: Path) -> dict[str, Any]:
    lock = load_lock(lock_path)
    if not torch.cuda.is_available():
        raise GateError("CUDA is unavailable")
    results = {"fp32": run_one(torch.float32), "fp16": run_one(torch.float16), "bf16": run_one(torch.bfloat16)}
    fp32 = results["fp32"]
    fp32_reference_error = float((fp32["output"] - fp32["reference_output"]).abs().max())
    checks = {
        "fp32_reference_output": fp32_reference_error <= 2e-5,
        "fp32_reference_gradient": fp32["reference_gradient_relative_l2"] <= 2e-5,
        "fp16_output": torch.allclose(results["fp16"]["output"], fp32["output"], atol=5e-3, rtol=5e-3),
        "bf16_output": torch.allclose(results["bf16"]["output"], fp32["output"], atol=2e-2, rtol=2e-2),
        "fp16_row_sum": results["fp16"]["row_sum_error"] <= 2e-3,
        "bf16_row_sum": results["bf16"]["row_sum_error"] <= 5e-3,
        "fp16_grad": max(relative_l2(a, b) for a, b in zip(results["fp16"]["gradients"], fp32["gradients"])) <= .02,
        "bf16_grad": max(relative_l2(a, b) for a, b in zip(results["bf16"]["gradients"], fp32["gradients"])) <= .05,
        "invalid_exact_zero": all(results[name]["invalid_gradient_exact_zero"] for name in results),
        "finite": all(results[name]["finite"] for name in results),
        "replicated_collapse": all(results[name]["replicated_collapse_output_delta"] <= (2e-5 if name == "fp32" else 2e-2) and results[name]["replicated_collapse_greedy_equal"] for name in results),
        "m1_control": all(results[name]["m1_output_delta"] <= (2e-5 if name == "fp32" else 2e-2) and results[name]["m1_greedy_equal"] for name in results),
    }
    activation_floor = max(
        2e-5,
        10.0 * max(
            max(results[name]["replicated_collapse_output_delta"], results[name]["m1_output_delta"])
            for name in results
        ),
    )
    serializable = {
        "schema_version": 1, "run_uid": lock["run_uid"], "cuda_device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "checks": checks,
        "metrics": {
            name: {k: v for k, v in value.items() if k not in {"output", "probability", "gradients", "reference_output"}}
            for name, value in results.items()
        },
        "fp32_reference_max_abs_error": fp32_reference_error,
        "synthetic_activation_null_floor": activation_floor,
        "status": "GO" if all(checks.values()) else "GPU_ENGINEERING_BLOCKED",
    }
    atomic_json(output_path, serializable)
    if not all(checks.values()):
        raise GateError(f"GPU numerical gate failed: {checks}")
    return serializable


def training_config(
    lock: dict[str, Any], seed: int, arm: str, output: Path,
    *, examples: int = 2048, optimizer_steps: int = 64,
) -> dict[str, Any]:
    sidecar_key = "training_alignment_sidecar_2048"
    sidecar = lock.get("assets", {}).get(sidecar_key, {}).get("container_path")
    if not sidecar:
        raise GateError("run lock does not bind a frozen training alignment sidecar")
    return {
        "model": {
            "base_model": "Qwen/Qwen3-0.6B", "teacher_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "is_do_alignment": True, "alignment_strategy": "soft_span_overlap_v2", "soft_alignment_top_k": 4,
            "soft_alignment_score_mode": "uniform", "soft_alignment_boundary_bonus": .5,
            "soft_alignment_boundary_tolerance": 1, "soft_alignment_min_weight": 0.0,
            "soft_alignment_confidence_mode": "entropy", "soft_alignment_confidence_alpha": .5,
            "soft_alignment_confidence_floor": .5, "soft_alignment_fallback_confidence": .25,
            "fpct_alignment_sanitizer": "certified_slot0_v1", "fpct_operator": arm,
            "include_response": False, "mapping": "last_aligned",
            "projector": {"type": "C2CProjector", "params": {"hidden_dim": 1024, "intermediate_dim": 1024, "num_layers": 3, "dropout": .1, "initial_temperature": 1.0, "final_temperature": .001, "anneal_steps": 64, "alignment_confidence_gate_mode": "token_mlp", "alignment_confidence_max_delta": 2.0}}
        },
        "training": {"learning_rate": 1e-4, "weight_decay": .01, "num_epochs": 1, "max_length": 1024, "device": "cuda", "scheduler_type": "linear", "warmup_ratio": .1, "max_grad_norm": 1.0, "gradient_accumulation_steps": 16, "per_device_train_batch_size": 1, "num_processes": 2, "freeze": ["teacher", "base"], "seed": seed, "fpct_formal_run": True, "expected_optimizer_steps": optimizer_steps, "fpct_expected_training_examples": examples, "fpct_alignment_cache_path": sidecar},
        "output": {"output_dir": str(output), "save_steps": 32, "eval_steps": 1000000, "wandb_config": {"project": "FPCT", "mode": "offline", "entity": "nics-efc", "run_name": f"fpct-{lock['run_uid']}-{seed}-{arm}"}},
        "data": {"type": "MMLUChatDataset", "kwargs": {"split": "auxiliary_train", "num_samples": examples, "max_word_count": 1024}, "train_ratio": 1.0, "split_mode": "seeded"}
    }


def run_arm(
    repo: Path, lock: dict[str, Any], seed: int, arm: str, output: Path,
    *, examples: int = 2048, optimizer_steps: int = 64,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "formal_config.json"
    atomic_json(config_path, training_config(lock, seed, arm, output, examples=examples, optimizer_steps=optimizer_steps))
    python = Path(sys.executable).resolve()
    bootstrap = repo / "script/runtime/fpct_bootstrap.py"
    target = repo / "script/train/SFT_train.py"
    attestation = output / "runtime_attestation_rank_{rank}.json"
    command = [
        str(python), "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", "--no-python",
        str(python), "-I", str(bootstrap), "--repo-root", str(repo), "--target", str(target),
        "--include-gpu-closure", "--attestation-out", str(attestation), "--", "--config", str(config_path),
    ]
    env = dict(os.environ); env.update({"FPCT_FORMAL_RUN": "1", "WANDB_MODE": "offline", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    started = time.time()
    subprocess.run(command, cwd=repo, env=env, check=True)
    manifest = output / "fpct_formal_integrity.json"
    if not manifest.is_file():
        raise GateError(f"training did not produce integrity manifest: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["wall_seconds"] = time.time() - started
    return payload


def train_triplet(lock_path: Path, seed: int, output_root: Path) -> dict[str, Any]:
    lock = load_lock(lock_path)
    if seed not in ARM_ORDER:
        raise GateError("formal seed must be 45..56")
    repo = Path(__file__).resolve().parents[2]
    triplet = output_root / f"seed_{seed}"
    triplet.mkdir(parents=True, exist_ok=False)
    records = {}
    for arm in ARM_ORDER[seed]:
        records[arm] = run_arm(repo, lock, seed, arm, triplet / arm)
    init_hashes = {records[arm]["step0_trainable_sha256"] for arm in records}
    data_hashes = {records[arm]["data_order_sha256"] for arm in records}
    keys_hashes = {records[arm]["trainable_keys_sha256"] for arm in records}
    status = "COMPLETE" if len(init_hashes) == len(data_hashes) == len(keys_hashes) == 1 else "INTEGRITY_FAILURE"
    payload = {"schema_version": 1, "run_uid": lock["run_uid"], "seed": seed, "arm_order": list(ARM_ORDER[seed]), "arms": records, "status": status}
    atomic_json(triplet / "triplet_manifest.json", payload)
    if status != "COMPLETE":
        raise GateError("matched triplet identity failure")
    return payload


def _to_device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "input_ids": [value.to(device) for value in batch["input_ids"]],
        "attention_mask": [value.to(device) for value in batch["attention_mask"]],
        "position_ids": batch["position_ids"].to(device),
        "labels": batch["labels"].to(device),
        "kv_cache_index": [value.to(device) for value in batch["kv_cache_index"]],
        "soft_alignment": [{key: value.to(device) for key, value in section.items()} for section in batch["soft_alignment"]],
    }


def _forward(model: Any, batch: dict[str, Any]) -> Any:
    return model.forward(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"], labels=batch["labels"],
        kv_cache_index=batch["kv_cache_index"], soft_alignment=batch["soft_alignment"], use_cache=True,
    )


def pretrained_smoke(lock_path: Path, gpu_gate_path: Path, output_path: Path) -> dict[str, Any]:
    lock = load_lock(lock_path)
    gate = json.loads(gpu_gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "GO":
        raise GateError("pretrained smoke requires GPU numerical GO")
    if not torch.cuda.is_available():
        raise GateError("CUDA unavailable")
    sft = loaded_module("sft_train")
    device = torch.device("cuda:0")
    config = training_config(lock, 104729, "c_post", output_path.parent / "unused", examples=128, optimizer_steps=4)
    model, receiver_tokenizer, aligner, sender_tokenizer = sft.setup_models(config["model"], "rosetta", str(device), torch.bfloat16)
    model.eval()
    prompts = [
        "Choose the best answer. Which process converts liquid water into vapor?\nA. freezing\nB. evaporation\nC. condensation\nD. deposition",
        "Choose the best answer. A triangle has angles 50, 60, and what remaining angle?\nA. 60\nB. 70\nC. 80\nD. 90",
        "Choose the best answer. Which word is closest in meaning to rapid?\nA. slow\nB. quick\nC. quiet\nD. heavy",
    ]
    messages = [[{"role": "user", "content": prompt}, {"role": "assistant", "content": "The correct answer is"}] for prompt in prompts]
    dataset = sft.AlignedChatDataset(messages, aligner, max_length=1024, soft_alignment_top_k=4, fpct_alignment_sanitizer="certified_slot0_v1")
    collator = sft.RosettaDataCollator(slm_tokenizer=receiver_tokenizer, llm_tokenizer=sender_tokenizer, max_length=1024, aligner=aligner, do_alignment=True)
    selected = None
    for item in dataset:
        if bool((item["soft_alignment"]["source_weights"] > 0).sum(-1).ge(2).any()):
            selected = item
            break
    if selected is None:
        raise GateError("fixed unlabeled prompts did not activate certified ambiguity")
    batch = _to_device_batch(collator([selected]), device)

    def timed(operator: str, *, replicated: bool = False, m1: bool = False, repeats: int = 1):
        local = batch
        if m1:
            local = dict(batch); local["soft_alignment"] = []
            for section in batch["soft_alignment"]:
                indices = section["source_indices"].clone(); weights = section["source_weights"].clone()
                indices[..., 1:] = -1; weights[..., 1:] = 0
                weights[..., 0] = torch.where(indices[..., 0] >= 0, torch.ones_like(weights[..., 0]), torch.zeros_like(weights[..., 0]))
                copy = dict(section); copy["source_indices"] = indices; copy["source_weights"] = weights; local["soft_alignment"].append(copy)
        model.fpct_operator = operator; model.fpct_replicated_collapse = replicated; model.fpct_instrumentation = operator == "f" and not replicated
        with torch.no_grad(): _forward(model, local)
        durations = []
        result = None
        torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            torch.cuda.synchronize(); start = time.perf_counter()
            with torch.no_grad(): result = _forward(model, local)
            torch.cuda.synchronize(); durations.append(time.perf_counter() - start)
        assert result is not None
        return result.logits.detach().float(), durations, torch.cuda.max_memory_allocated() / 2**30, {key: float(value.detach().cpu()) for key, value in model._fpct_mechanism_metrics.items()}

    cpost, cpost_times, cpost_hbm, _ = timed("c_post", repeats=7)
    factorized, f_times, f_hbm, diagnostics = timed("f", repeats=7)
    replicated, _, _, _ = timed("f", replicated=True)
    m1_post, _, _, _ = timed("c_post", m1=True)
    m1_f, _, _, _ = timed("f", m1=True)
    cpost_median = statistics.median(cpost_times)
    f_median = statistics.median(f_times)
    cpost_p95 = sorted(cpost_times)[-1]
    f_p95 = sorted(f_times)[-1]
    latency_median_ratio = f_median / cpost_median
    latency_p95_ratio = f_p95 / cpost_p95

    model.fpct_operator = "f"; model.fpct_replicated_collapse = False
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        with torch.no_grad(): _forward(model, batch)
    host_sync_events = sorted({event.key for event in profile.key_averages() if "cudaDeviceSynchronize" in event.key or "cudaStreamSynchronize" in event.key})
    delta = float((factorized - cpost).abs().max().cpu())
    replicated_delta = float((replicated - cpost).abs().max().cpu())
    m1_delta = float((m1_f - m1_post).abs().max().cpu())
    floor = float(gate["synthetic_activation_null_floor"])
    activated = delta > floor and any(diagnostics.get(key, 0.0) > floor for key in ("gamma_kl_prior", "jensen_gap", "gamma_query_variance", "candidate_logit_range"))
    checks = {
        "activation": activated,
        "replicated_collapse": replicated_delta <= floor,
        "m1_control": m1_delta <= floor,
        "finite": bool(torch.isfinite(factorized).all()),
        "hbm": max(cpost_hbm, f_hbm) < .9 * torch.cuda.get_device_properties(0).total_memory / 2**30,
        "latency_median_ratio": latency_median_ratio <= 1.50,
        "latency_p95_ratio": latency_p95_ratio <= 1.75,
        "no_profiled_host_sync": not host_sync_events,
    }
    payload = {
        "schema_version": 1, "run_uid": lock["run_uid"], "checks": checks,
        "activation_floor": floor, "max_output_delta": delta,
        "replicated_delta": replicated_delta, "m1_delta": m1_delta,
        "latency_seconds": {
            "c_post_samples": cpost_times, "f_samples": f_times,
            "c_post_median": cpost_median, "f_median": f_median,
            "c_post_p95": cpost_p95, "f_p95": f_p95,
            "median_ratio": latency_median_ratio, "p95_ratio": latency_p95_ratio,
        },
        "peak_hbm_gib": {"c_post": cpost_hbm, "f": f_hbm}, "diagnostics": diagnostics,
        "profiled_host_sync_events": host_sync_events,
        "status": "GO" if all(checks.values()) else "GPU_ENGINEERING_BLOCKED",
    }
    atomic_json(output_path, payload)
    if payload["status"] != "GO": raise GateError(f"pretrained activation smoke failed: {checks}")
    return payload


def matched_smoke(lock_path: Path, output_root: Path) -> dict[str, Any]:
    lock = load_lock(lock_path); repo = Path(__file__).resolve().parents[2]
    root = output_root / "diagnostic_seed_104729"; root.mkdir(parents=True, exist_ok=False)
    records = {arm: run_arm(repo, lock, 104729, arm, root / arm, examples=128, optimizer_steps=4) for arm in ("c_pre", "c_post", "f")}
    checks = {
        "step0_identity": len({record["step0_trainable_sha256"] for record in records.values()}) == 1,
        "trainable_keys": len({record["trainable_keys_sha256"] for record in records.values()}) == 1,
        "data_order": len({record["data_order_sha256"] for record in records.values()}) == 1,
        "four_steps": all(record["optimizer_steps"] == 4 for record in records.values()),
    }
    payload = {"schema_version": 1, "run_uid": lock["run_uid"], "seed": 104729, "arms": records, "checks": checks, "status": "GO" if all(checks.values()) else "INTEGRITY_FAILURE"}
    atomic_json(root / "matched_smoke_manifest.json", payload)
    if payload["status"] != "GO": raise GateError(f"matched smoke failed: {checks}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe"); probe.add_argument("--run-lock", type=Path, required=True)
    gpu = sub.add_parser("gpu-numerical"); gpu.add_argument("--run-lock", type=Path, required=True); gpu.add_argument("--output", type=Path, required=True)
    pretrained = sub.add_parser("pretrained-smoke"); pretrained.add_argument("--run-lock", type=Path, required=True); pretrained.add_argument("--gpu-gate", type=Path, required=True); pretrained.add_argument("--output", type=Path, required=True)
    matched = sub.add_parser("matched-smoke"); matched.add_argument("--run-lock", type=Path, required=True); matched.add_argument("--output-root", type=Path, required=True)
    train = sub.add_parser("train-triplet"); train.add_argument("--run-lock", type=Path, required=True); train.add_argument("--seed", type=int, required=True); train.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "probe": payload = {"status": "SEALED", "run_uid": load_lock(args.run_lock)["run_uid"]}
    elif args.command == "gpu-numerical": payload = gpu_numerical(args.run_lock, args.output)
    elif args.command == "pretrained-smoke": payload = pretrained_smoke(args.run_lock, args.gpu_gate, args.output)
    elif args.command == "matched-smoke": payload = matched_smoke(args.run_lock, args.output_root)
    else: payload = train_triplet(args.run_lock, args.seed, args.output_root)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
