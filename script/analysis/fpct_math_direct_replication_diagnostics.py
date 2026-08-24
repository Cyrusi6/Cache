#!/usr/bin/env python3
"""Low-cost gradient and teacher-forced diagnostics for math-direct replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


TASK_COUNTS = {"ai2-arc": 128, "openbookqa": 70, "mmlu-redux": 128}
PROTOCOL_ID = "fpct_math_direct_replication_v1"


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _hash_value(digest: Any, value: Any, prefix: str = "root") -> None:
    import torch

    digest.update(prefix.encode("utf-8") + b"\0")
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value):
            _hash_value(digest, value[key], f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _hash_value(digest, child, f"{prefix}[{index}]")
    elif value is not None:
        digest.update(repr(value).encode("utf-8"))


def batch_sha256(batch: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, batch)
    return digest.hexdigest()


def _rng_snapshot() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(value: dict[str, Any]) -> None:
    import torch

    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state_all(value["cuda"])


def _parameter_group(name: str) -> str:
    leaf = name.split("projector_list.", 1)[-1]
    if any(token in leaf for token in ("gate", "confidence", "entropy", "temperature")):
        return "nuisance_gate_confidence"
    if ".key_" in f".{leaf}" or leaf.startswith("key_"):
        return "key_path"
    if ".value_" in f".{leaf}" or leaf.startswith("value_"):
        return "value_path"
    return "other"


def _layer(name: str) -> int:
    match = re.search(r"(?:^|\.)projector_list\.(\d+)\.", name)
    if match is None:
        raise ValueError(f"trainable parameter is not a projector parameter: {name}")
    return int(match.group(1))


def gradient_comparison(
    c_post: dict[str, Any | None], f_grad: dict[str, Any | None]
) -> dict[str, Any]:
    import torch

    if set(c_post) != set(f_grad):
        raise ValueError("gradient parameter universes differ")
    buckets: dict[tuple[int, str], list[float]] = {}
    global_bucket = [0.0, 0.0, 0.0, 0, 0]
    for name in sorted(c_post):
        left = c_post[name]
        right = f_grad[name]
        if left is not None and not bool(torch.isfinite(left).all()):
            raise ValueError(f"nonfinite C_post gradient: {name}")
        if right is not None and not bool(torch.isfinite(right).all()):
            raise ValueError(f"nonfinite F gradient: {name}")
        l2 = float((left.double() * left.double()).sum()) if left is not None else 0.0
        r2 = float((right.double() * right.double()).sum()) if right is not None else 0.0
        dot = (
            float((left.double() * right.double()).sum())
            if left is not None and right is not None
            else 0.0
        )
        values = buckets.setdefault((_layer(name), _parameter_group(name)), [0.0] * 5)
        values[0] += l2
        values[1] += r2
        values[2] += dot
        values[3] += int(left is None)
        values[4] += int(right is None)
        global_bucket[0] += l2
        global_bucket[1] += r2
        global_bucket[2] += dot
        global_bucket[3] += int(left is None)
        global_bucket[4] += int(right is None)

    def finalize(values: list[float]) -> dict[str, Any]:
        left_norm = math.sqrt(values[0])
        right_norm = math.sqrt(values[1])
        cosine = values[2] / (left_norm * right_norm) if left_norm and right_norm else None
        return {
            "c_post_norm": left_norm,
            "f_norm": right_norm,
            "cosine": cosine,
            "norm_ratio_f_over_c_post": right_norm / left_norm if left_norm else None,
            "c_post_missing_parameter_count": int(values[3]),
            "f_missing_parameter_count": int(values[4]),
        }

    rows = [
        {"layer": layer, "group": group, **finalize(values)}
        for (layer, group), values in sorted(buckets.items())
    ]
    return {"global": finalize(global_bucket), "layer_groups": rows}


def run_step0_gradient(config_path: Path, output: Path, device: str) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from rosetta.train.dataset_adapters import (
        AlignedChatDataset,
        RosettaDataCollator,
        create_dataset,
    )
    from script.train.SFT_train import (
        _create_dataset_split,
        enable_full_determinism,
        freeze_model,
        load_config,
        set_seed,
        setup_models,
        train_step,
        unfreeze_projectors,
    )

    config = load_config(str(config_path))
    model_config = config["model"]
    training = config["training"]
    data = config["data"]
    seed = int(training["seed"])
    set_seed(seed)
    enable_full_determinism()
    model, tokenizer, aligner, sender_tokenizer = setup_models(
        model_config, "rosetta", device, torch.bfloat16
    )
    freeze_model(model.model_list[0])
    freeze_model(model.model_list[1])
    unfreeze_projectors(model)

    source = create_dataset(dataset_type=data["type"], **data["kwargs"])
    full = AlignedChatDataset(
        source,
        aligner,
        max_length=training.get("max_length", 2048),
        soft_alignment_top_k=model_config.get("soft_alignment_top_k", 4),
        fpct_alignment_sanitizer=model_config.get("fpct_alignment_sanitizer", "none"),
        fpct_alignment_cache_path=training.get("fpct_alignment_cache_path"),
    )
    train, _evaluation, split_mode = _create_dataset_split(
        full,
        len(full),
        0,
        seed=seed,
        split_mode=data.get("split_mode", "seeded"),
    )
    sampler = DistributedSampler(
        train, num_replicas=2, rank=0, shuffle=True, seed=seed, drop_last=False
    )
    sampler.set_epoch(0)
    order = list(iter(sampler))
    collator = RosettaDataCollator(
        slm_tokenizer=tokenizer,
        llm_tokenizer=sender_tokenizer,
        max_length=training.get("max_length", 2048),
        aligner=aligner,
        do_alignment=model_config.get("is_do_alignment", False),
    )
    loader = DataLoader(train, batch_size=1, sampler=sampler, collate_fn=collator)
    batch = next(iter(loader))
    fixed_batch_sha = batch_sha256(batch)
    rng = _rng_snapshot()
    model.train()
    gradients: dict[str, dict[str, Any | None]] = {}
    losses: dict[str, float] = {}
    keys = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    for operator in ("c_post", "f"):
        model.zero_grad(set_to_none=True)
        _restore_rng(rng)
        model.fpct_operator = operator
        model.set_fpct_centered_lambda(1.0)
        loss = train_step(
            model,
            batch,
            tokenizer,
            training.get("max_length", 2048),
            device,
            "rosetta",
            include_auxiliary_loss=True,
        )
        if not bool(torch.isfinite(loss.detach())):
            raise RuntimeError(f"nonfinite step-0 loss for {operator}")
        loss.backward()
        losses[operator] = float(loss.detach().float().cpu())
        gradients[operator] = {
            name: None if parameter.grad is None else parameter.grad.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        del loss
        torch.cuda.empty_cache()

    comparison = gradient_comparison(gradients["c_post"], gradients["f"])
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "GO",
        "seed": seed,
        "split_mode": split_mode,
        "rank": 0,
        "world_size_contract": 2,
        "sampler_position": int(order[0]),
        "source_dataset_index": int(train.indices[order[0]]),
        "batch_sha256": fixed_batch_sha,
        "trainable_parameter_count": len(keys),
        "loss": losses,
        "loss_delta_f_minus_c_post": losses["f"] - losses["c_post"],
        "gradients": comparison,
        "optimizer_step_performed": False,
    }
    atomic_json(output, result)
    return result


def _load_items(input_cache: Path, expected_sha: str, task: str) -> list[dict[str, Any]]:
    import torch

    if sha256_file(input_cache) != expected_sha:
        raise ValueError("compact E0-design input SHA mismatch")
    value = torch.load(input_cache, map_location="cpu", weights_only=False)
    if (
        value.get("split_role") != "e0_design"
        or value.get("e1_pilot_consumed") is not False
        or value.get("model_or_checkpoint_loaded") is not False
    ):
        raise ValueError("compact input firewall/provenance mismatch")
    items = value.get("items")
    counts = Counter(str(item.get("task")) for item in items)
    if dict(counts) != TASK_COUNTS:
        raise ValueError("compact input task counts mismatch")
    selected = [item for item in items if item["task"] == task]
    return sorted(selected, key=lambda row: row["sample_sha256"])


def run_teacher_forced(
    config_path: Path,
    checkpoint: Path,
    checkpoint_arm: str,
    seed: int,
    task: str,
    input_cache: Path,
    input_sha256: str,
    output: Path,
    device_name: str,
) -> dict[str, Any]:
    import torch
    import yaml
    from transformers import AutoTokenizer
    from rosetta.train.dataset_adapters import RosettaDataCollator
    from rosetta.utils.evaluate import load_rosetta_model, set_default_chat_template
    from rosetta.utils.model_loading import resolve_model_path
    from script.experiment.fpct_e1_fast_rca import (
        _active_section_labels,
        _gold_logp,
        _move_to_device,
    )

    if checkpoint_arm not in {"c_post", "f"} or task not in TASK_COUNTS:
        raise ValueError("invalid checkpoint arm or task")
    if output.exists():
        raise FileExistsError(output)
    items = _load_items(input_cache, input_sha256, task)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model"]["rosetta_config"]["checkpoints_dir"] = str(checkpoint)
    config["model"]["rosetta_config"]["fpct_operator"] = "c_post"
    config["model"]["rosetta_config"]["fpct_instrumentation"] = False
    config["eval"]["gpu_ids"] = [0]
    device = torch.device(device_name)
    model, receiver_tokenizer = load_rosetta_model(
        config["model"], config["eval"], device, config["model"].get("generation_config", {})
    )
    model.eval()
    sender_name = config["model"]["rosetta_config"]["teacher_model"]
    sender_tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(sender_name))
    set_default_chat_template(sender_tokenizer, sender_name)
    collator = RosettaDataCollator(
        receiver_tokenizer,
        sender_tokenizer,
        max_length=32768,
        aligner=None,
        do_alignment=False,
    )
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    rows_path = temporary / "rows.jsonl"
    row_count = 0
    with rows_path.open("x", encoding="utf-8") as handle:
        for item in items:
            for operator in ("c_post", "f"):
                model.fpct_operator = operator
                model.set_fpct_centered_lambda(1.0)
                batch = collator([item["feature"]])
                labels = batch.pop("labels").to(device)
                batch = _move_to_device(batch, device)
                active_labels = _active_section_labels(labels, batch.get("kv_cache_index"))
                with torch.no_grad():
                    outputs = model(**batch, labels=labels, use_cache=True)
                cell = {
                    ("c_post", "c_post"): "Y_CC",
                    ("c_post", "f"): "Y_CF",
                    ("f", "c_post"): "Y_FC",
                    ("f", "f"): "Y_FF",
                }[(checkpoint_arm, operator)]
                row = {
                    "schema_version": 1,
                    "protocol_id": PROTOCOL_ID,
                    "seed": seed,
                    "checkpoint_arm": checkpoint_arm,
                    "inference_operator": operator,
                    "cell": cell,
                    "task": task,
                    "sample_sha256": item["sample_sha256"],
                    "content_group_sha256": item["content_group_sha256"],
                    **_gold_logp(outputs, active_labels),
                }
                handle.write(canonical_json(row).decode("utf-8"))
                row_count += 1
                del outputs, batch, labels, active_labels
    receipt = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "seed": seed,
        "checkpoint_arm": checkpoint_arm,
        "task": task,
        "sample_count": len(items),
        "row_count": row_count,
        "rows_sha256": sha256_file(rows_path),
        "input_cache_sha256": input_sha256,
        "e1_pilot_consumed": False,
        "model_selection_consumed": False,
        "test_consumed": False,
    }
    (temporary / "receipt.json").write_bytes(canonical_json(receipt))
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    gradient = sub.add_parser("step0-gradient")
    gradient.add_argument("--config", type=Path, required=True)
    gradient.add_argument("--output", type=Path, required=True)
    gradient.add_argument("--device", default="cuda:0")
    teacher = sub.add_parser("teacher-forced")
    teacher.add_argument("--config", type=Path, required=True)
    teacher.add_argument("--checkpoint", type=Path, required=True)
    teacher.add_argument("--checkpoint-arm", choices=("c_post", "f"), required=True)
    teacher.add_argument("--seed", type=int, required=True)
    teacher.add_argument("--task", choices=tuple(TASK_COUNTS), required=True)
    teacher.add_argument("--input-cache", type=Path, required=True)
    teacher.add_argument("--input-sha256", required=True)
    teacher.add_argument("--output", type=Path, required=True)
    teacher.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "step0-gradient":
        result = run_step0_gradient(args.config, args.output, args.device)
    else:
        result = run_teacher_forced(
            args.config,
            args.checkpoint,
            args.checkpoint_arm,
            args.seed,
            args.task,
            args.input_cache,
            args.input_sha256,
            args.output,
            args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
