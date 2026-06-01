from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from rosetta.model.aligner import AlignmentStrategy, TokenAligner
from rosetta.train.dataset_adapters import (
    AlignedChatDataset,
    RosettaDataCollator,
    create_dataset,
)
from rosetta.utils.evaluate import load_rosetta_model, set_default_chat_template


Accumulator = MutableMapping[str, float]


def _new_accumulator() -> Accumulator:
    return defaultdict(float)


def _update_stats(
    stats: Accumulator,
    key_delta: torch.Tensor,
    value_delta: torch.Tensor,
    key_confidence: torch.Tensor,
    value_confidence: torch.Tensor,
    source_confidence: torch.Tensor,
    entropy: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    mask_expanded = torch.broadcast_to(mask, key_delta.shape).to(dtype=key_delta.dtype)
    count = float(mask_expanded.sum().item())
    if count <= 0:
        return

    source_confidence = torch.broadcast_to(source_confidence, key_delta.shape)
    entropy = torch.broadcast_to(entropy, key_delta.shape)
    stats["count"] += count
    stats["key_delta_sum"] += float((key_delta * mask_expanded).sum().item())
    stats["value_delta_sum"] += float((value_delta * mask_expanded).sum().item())
    stats["key_delta_abs_sum"] += float((key_delta.abs() * mask_expanded).sum().item())
    stats["value_delta_abs_sum"] += float(
        (value_delta.abs() * mask_expanded).sum().item()
    )
    stats["key_confidence_sum"] += float(
        (key_confidence * mask_expanded).sum().item()
    )
    stats["value_confidence_sum"] += float(
        (value_confidence * mask_expanded).sum().item()
    )
    stats["source_confidence_sum"] += float(
        (source_confidence * mask_expanded).sum().item()
    )
    stats["entropy_sum"] += float((entropy * mask_expanded).sum().item())
    stats["key_delta_abs_max"] = max(
        stats["key_delta_abs_max"],
        float(key_delta.abs().masked_select(mask_expanded.bool()).max().item()),
    )
    stats["value_delta_abs_max"] = max(
        stats["value_delta_abs_max"],
        float(value_delta.abs().masked_select(mask_expanded.bool()).max().item()),
    )


def _finalize_stats(name_parts: Dict[str, Any], stats: Accumulator) -> Dict[str, Any]:
    count = stats.get("count", 0.0)
    row: Dict[str, Any] = dict(name_parts)
    row["count"] = int(count)
    if count <= 0:
        for key in [
            "key_delta_mean",
            "value_delta_mean",
            "key_delta_abs_mean",
            "value_delta_abs_mean",
            "key_confidence_mean",
            "value_confidence_mean",
            "source_confidence_mean",
            "entropy_mean",
            "key_delta_abs_max",
            "value_delta_abs_max",
        ]:
            row[key] = 0.0
        return row

    row["key_delta_mean"] = stats["key_delta_sum"] / count
    row["value_delta_mean"] = stats["value_delta_sum"] / count
    row["key_delta_abs_mean"] = stats["key_delta_abs_sum"] / count
    row["value_delta_abs_mean"] = stats["value_delta_abs_sum"] / count
    row["key_confidence_mean"] = stats["key_confidence_sum"] / count
    row["value_confidence_mean"] = stats["value_confidence_sum"] / count
    row["source_confidence_mean"] = stats["source_confidence_sum"] / count
    row["entropy_mean"] = stats["entropy_sum"] / count
    row["key_delta_abs_max"] = stats["key_delta_abs_max"]
    row["value_delta_abs_max"] = stats["value_delta_abs_max"]
    return row


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    input_ids = [ids.to(device) for ids in batch["input_ids"]]
    attention_mask = [mask.to(device) for mask in batch["attention_mask"]]
    soft_alignment = [
        {
            "source_indices": section["source_indices"].to(device),
            "source_weights": section["source_weights"].to(device),
            "source_confidence": section["source_confidence"].to(device),
        }
        for section in batch["soft_alignment"]
    ]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": batch["position_ids"].to(device),
        "labels": batch["labels"].to(device),
        "kv_cache_index": [section.to(device) for section in batch["kv_cache_index"]],
        "soft_alignment": soft_alignment,
    }


def _clear_projector_records(projectors: Iterable[Any]) -> None:
    for projector in projectors:
        if hasattr(projector, "alignment_diagnostic_records"):
            projector.alignment_diagnostic_records.clear()


def _iter_projector_records(projectors: Iterable[Any]):
    for layer_idx, projector in enumerate(projectors):
        for record in getattr(projector, "alignment_diagnostic_records", []):
            yield layer_idx, record


def _bucket_masks(
    source_confidence: torch.Tensor,
    entropy: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    fallback = source_confidence <= 0.26
    ambiguous = (entropy > 1e-6) & (~fallback)
    low_confidence = (source_confidence < 0.999) & (entropy <= 1e-6) & (~fallback)
    confident = (source_confidence >= 0.999) & (entropy <= 1e-6)
    return {
        "all": torch.ones_like(source_confidence, dtype=torch.bool),
        "confident_1to1": confident,
        "entropy_ambiguous": ambiguous,
        "fallback": fallback,
        "low_confidence_nonentropy": low_confidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose route-1 token_mlp confidence gates by layer/head/bucket."
    )
    parser.add_argument(
        "--eval-config",
        default=(
            "local/tmp/eval_configs/route1_alignment_v22/"
            "route1_v22_qwen3_tinyllama_token_mlp_entropy050_small2048_mmlu-redux.yaml"
        ),
    )
    parser.add_argument("--dataset-type", default="MMLUChatDataset")
    parser.add_argument("--dataset-split", default="auxiliary_train")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        default=(
            "local/final_results/route1_alignment_v25/diagnostics/"
            "v22_token_mlp_gate_mmlu_aux64"
        ),
    )
    args = parser.parse_args()

    with open(args.eval_config, "r", encoding="utf-8") as handle:
        eval_yaml = yaml.safe_load(handle)

    model_config = eval_yaml["model"]
    rosetta_config = model_config["rosetta_config"]
    eval_config = eval_yaml.get("eval", {})
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, slm_tokenizer = load_rosetta_model(
        model_config=model_config,
        eval_config=eval_config,
        device=device,
        generation_config=model_config.get("generation_config"),
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(rosetta_config["teacher_model"])
    set_default_chat_template(llm_tokenizer, rosetta_config["teacher_model"])

    for projector in model.projector_list:
        projector.capture_alignment_diagnostics = True

    aligner = TokenAligner(
        slm_tokenizer=slm_tokenizer,
        llm_tokenizer=llm_tokenizer,
        strategy=AlignmentStrategy(rosetta_config["alignment_strategy"]),
        soft_alignment_score_mode=rosetta_config.get(
            "soft_alignment_score_mode", "overlap"
        ),
        soft_alignment_boundary_bonus=rosetta_config.get(
            "soft_alignment_boundary_bonus", 0.0
        ),
        soft_alignment_boundary_tolerance=rosetta_config.get(
            "soft_alignment_boundary_tolerance", 1
        ),
        soft_alignment_min_weight=rosetta_config.get("soft_alignment_min_weight", 0.0),
        soft_alignment_confidence_mode=rosetta_config.get(
            "soft_alignment_confidence_mode", "none"
        ),
        soft_alignment_confidence_alpha=rosetta_config.get(
            "soft_alignment_confidence_alpha", 0.5
        ),
        soft_alignment_confidence_floor=rosetta_config.get(
            "soft_alignment_confidence_floor", 0.0
        ),
        soft_alignment_fallback_confidence=rosetta_config.get(
            "soft_alignment_fallback_confidence", 1.0
        ),
    )

    raw_dataset = create_dataset(
        args.dataset_type,
        split=args.dataset_split,
        num_samples=args.num_samples,
        max_word_count=args.max_length,
    )
    aligned_dataset = AlignedChatDataset(
        raw_dataset,
        aligner=aligner,
        max_length=args.max_length,
        soft_alignment_top_k=int(rosetta_config.get("soft_alignment_top_k", 4)),
    )
    collator = RosettaDataCollator(
        slm_tokenizer=slm_tokenizer,
        llm_tokenizer=llm_tokenizer,
        max_length=args.max_length,
        aligner=aligner,
        do_alignment=True,
    )
    loader = DataLoader(
        aligned_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    layer_head_stats: Dict[tuple, Accumulator] = defaultdict(_new_accumulator)
    layer_bucket_stats: Dict[tuple, Accumulator] = defaultdict(_new_accumulator)
    global_stats: Accumulator = _new_accumulator()
    losses: List[float] = []
    batches = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batches += 1
            _clear_projector_records(model.projector_list)
            model_inputs = _move_batch_to_device(batch, device)
            outputs = model.forward(**model_inputs, use_cache=True)
            if outputs.loss is not None:
                losses.append(float(outputs.loss.detach().float().cpu().item()))

            for layer_idx, record in _iter_projector_records(model.projector_list):
                key_delta = record["key_delta"]
                value_delta = record["value_delta"]
                key_confidence = record["key_confidence"]
                value_confidence = record["value_confidence"]
                source_confidence = record["source_confidence"]
                entropy = record["entropy"]
                bucket_masks = _bucket_masks(source_confidence, entropy)

                _update_stats(
                    global_stats,
                    key_delta,
                    value_delta,
                    key_confidence,
                    value_confidence,
                    source_confidence,
                    entropy,
                    bucket_masks["all"],
                )
                for bucket_name, bucket_mask in bucket_masks.items():
                    _update_stats(
                        layer_bucket_stats[(layer_idx, bucket_name)],
                        key_delta,
                        value_delta,
                        key_confidence,
                        value_confidence,
                        source_confidence,
                        entropy,
                        bucket_mask,
                    )

                num_heads = key_delta.shape[1]
                all_mask = bucket_masks["all"]
                for head_idx in range(num_heads):
                    _update_stats(
                        layer_head_stats[(layer_idx, head_idx)],
                        key_delta[:, head_idx : head_idx + 1],
                        value_delta[:, head_idx : head_idx + 1],
                        key_confidence[:, head_idx : head_idx + 1],
                        value_confidence[:, head_idx : head_idx + 1],
                        source_confidence,
                        entropy,
                        all_mask,
                    )

    layer_head_rows = [
        _finalize_stats({"layer": layer, "head": head}, stats)
        for (layer, head), stats in sorted(layer_head_stats.items())
    ]
    layer_bucket_rows = [
        _finalize_stats({"layer": layer, "bucket": bucket}, stats)
        for (layer, bucket), stats in sorted(layer_bucket_stats.items())
    ]
    summary = _finalize_stats({"scope": "global"}, global_stats)
    summary.update(
        {
            "eval_config": args.eval_config,
            "checkpoint_dir": rosetta_config["checkpoints_dir"],
            "dataset_type": args.dataset_type,
            "dataset_split": args.dataset_split,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "batches": batches,
            "mean_loss": sum(losses) / len(losses) if losses else None,
            "top_layers_by_key_delta_abs": sorted(
                [
                    row
                    for row in layer_bucket_rows
                    if row["bucket"] == "all" and row["count"] > 0
                ],
                key=lambda row: row["key_delta_abs_mean"],
                reverse=True,
            )[:8],
            "top_layers_by_value_delta_abs": sorted(
                [
                    row
                    for row in layer_bucket_rows
                    if row["bucket"] == "all" and row["count"] > 0
                ],
                key=lambda row: row["value_delta_abs_mean"],
                reverse=True,
            )[:8],
        }
    )

    _write_csv(output_dir / "layer_head_stats.csv", layer_head_rows)
    _write_csv(output_dir / "layer_bucket_stats.csv", layer_bucket_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# Route-1 V2.2 Confidence Gate Diagnostics\n\n"
            f"- Eval config: `{args.eval_config}`\n"
            f"- Samples: `{args.num_samples}` from `{args.dataset_type}` "
            f"`{args.dataset_split}`\n"
            f"- Mean loss: `{summary['mean_loss']}`\n"
            f"- Global key delta abs mean: `{summary['key_delta_abs_mean']}`\n"
            f"- Global value delta abs mean: `{summary['value_delta_abs_mean']}`\n"
            f"- Layer/head stats: `layer_head_stats.csv`\n"
            f"- Layer/bucket stats: `layer_bucket_stats.csv`\n"
            f"- Machine summary: `summary.json`\n"
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
