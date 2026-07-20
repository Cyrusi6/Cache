from __future__ import annotations

"""Record old-execution provenance without tokenizer use or model forward."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
import transformers

from rosetta.model.projector import C2CProjector


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git_show(repo: Path, commit: str, relative: str) -> str:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"], cwd=repo, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            value = ast.get_source_segment(source, node)
            if value is None:
                break
            return value
    raise RuntimeError(f"function not found: {name}")


def first_source_weight_dtype(sidecar: Path) -> str:
    payload = torch.load(sidecar, map_location="cpu", weights_only=False)
    for item in payload.get("items", []):
        soft = item.get("soft_alignment", {})
        value = soft.get("source_weights")
        if isinstance(value, torch.Tensor):
            return str(value.dtype)
    raise RuntimeError("sidecar has no source_weights tensor")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--old-commit", required=True)
    parser.add_argument("--old-run-lock", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--receiver-config", type=Path, required=True)
    parser.add_argument("--sender-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    old_lock = json.loads(args.old_run_lock.read_text())
    receiver = json.loads(args.receiver_config.read_text())
    sender = json.loads(args.sender_config.read_text())
    runner_source = git_show(repo, args.old_commit, "script/experiment/fpct_confirmatory_runner.py")
    attention_source = git_show(repo, args.old_commit, "rosetta/model/fpct_attention.py")
    wrapper_source = git_show(repo, args.old_commit, "rosetta/model/wrapper.py")
    projector_source = git_show(repo, args.old_commit, "rosetta/model/projector.py")
    config_source = function_source(runner_source, "training_config")

    target_dim = int(receiver.get("head_dim") or receiver["hidden_size"] // receiver["num_attention_heads"])
    source_dim = int(sender.get("head_dim") or sender["hidden_size"] // sender["num_attention_heads"])
    prototype = C2CProjector(
        source_dim=source_dim,
        target_dim=target_dim,
        source_num_heads=int(sender["num_key_value_heads"]),
        target_num_heads=int(receiver["num_key_value_heads"]),
        hidden_dim=1024,
        intermediate_dim=1024,
        num_layers=3,
        dropout=0.1,
        dtype=torch.float32,
        alignment_confidence_gate_mode="token_mlp",
        alignment_confidence_max_delta=2.0,
    )
    layer_count = int(receiver["num_hidden_layers"])
    key_logit = float(prototype.key_gate_logit.detach())
    value_logit = float(prototype.value_gate_logit.detach())
    if key_logit != 0.0 or value_logit != 0.0:
        raise RuntimeError("fresh projector scalar gates are not exactly zero")

    module_files = {}
    for module in (sys.modules[__name__], sys.modules[C2CProjector.__module__]):
        path = Path(module.__file__).resolve()
        module_files[module.__name__] = {"path": str(path), "sha256": sha256(path)}

    payload = {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2_zero_output_provenance_v1",
        "status": "COMPLETE_NO_TOKENIZER_NO_MODEL_FORWARD",
        "old_execution": {
            "scientific_commit": args.old_commit,
            "image_reference": old_lock["image"]["reference"],
            "run_uid": old_lock["run_uid"],
            "run_lock_sha256": sha256(args.old_run_lock),
            "projector_checkpoint_dir": None,
            "projector_state": "fresh",
            "training_config_explicit_eager": '"attn_implementation"' in config_source,
            "backend_claim": "Old execution had no runtime proof of eager; under current automatic dispatch rules it may have used SDPA.",
        },
        "fresh_projector": {
            "mapped_layer_count": layer_count,
            "key_gate_logits": [key_logit] * layer_count,
            "value_gate_logits": [value_logit] * layer_count,
            "legacy_scalar_gate_eval_mode": prototype.legacy_scalar_gate_eval_mode,
            "alignment_confidence_eval_mode": prototype.alignment_confidence_eval_mode,
            "checkpoint_native_eval_expression_present": "(key_gate_logit > 0).float()" in projector_source,
            "expected_classification": "EXPECTED_NATIVE_NULL",
        },
        "old_prior_contract": {
            "frozen_sidecar_source_weights_dtype": first_source_weight_dtype(args.sidecar),
            "canonical_prior_dtype": "followed source cache dtype (BF16 in old smoke)",
            "log_prior_dtype": "followed packed layout prior dtype",
            "packed_mask_dtype": "cast to key/KV dtype before adding log prior",
            "kv_dtype": "BF16",
            "evidence": {
                "fpct_attention_source_sha256": hashlib.sha256(attention_source.encode()).hexdigest(),
                "wrapper_source_sha256": hashlib.sha256(wrapper_source.encode()).hexdigest(),
                "prior_cast_to_key_dtype_present": "dtype=key.dtype" in attention_source,
                "source_prior_cast_to_cache_dtype_present": "source_key_cache.dtype" in wrapper_source,
            },
        },
        "serialized_model_configs": {
            "receiver_architectures": receiver.get("architectures"),
            "sender_architectures": sender.get("architectures"),
            "receiver_attn_implementation": receiver.get("_attn_implementation"),
            "sender_attn_implementation": sender.get("_attn_implementation"),
            "receiver_expected_attention_class": "Qwen3Attention",
            "sender_expected_attention_class": "LlamaAttention",
            "actual_per_layer_runtime_attestation": "DEFERRED_TO_NEW_R2_IMAGE_BEFORE_FORWARD"
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "transformers": transformers.__version__,
            "module_files": module_files,
        },
        "assets": {
            "sidecar": {"path": str(args.sidecar.resolve()), "sha256": sha256(args.sidecar)},
            "receiver_config": {"path": str(args.receiver_config.resolve()), "sha256": sha256(args.receiver_config)},
            "sender_config": {"path": str(args.sender_config.resolve()), "sha256": sha256(args.sender_config)},
        },
        "firewall": {
            "tokenizer_loaded": False,
            "natural_prompt_read": False,
            "model_weight_loaded": False,
            "model_forward": False,
            "accuracy_read": False,
        },
    }
    if payload["old_execution"]["training_config_explicit_eager"]:
        raise RuntimeError("old training config unexpectedly contained explicit eager")
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
