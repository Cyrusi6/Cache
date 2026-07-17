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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from rosetta.model.aligner import AlignmentStrategy, TokenAligner
from rosetta.train.dataset_adapters import (
    AlignedChatDataset,
    RosettaDataCollator,
    create_dataset,
)
from rosetta.utils.evaluate import load_rosetta_model, set_default_chat_template
from rosetta.utils.gate_diagnostics import (
    GATE_DEFINITION,
    GateDiagnosticsAccumulator,
    clear_projector_gate_diagnostic_records,
    configure_projector_gate_diagnostics,
)
from rosetta.utils.model_loading import resolve_model_path


Accumulator = MutableMapping[str, float]


def _load_teacher_tokenizer(model_id: str):
    resolved_path = resolve_model_path(model_id)
    tokenizer = AutoTokenizer.from_pretrained(resolved_path)
    set_default_chat_template(tokenizer, model_id)
    return tokenizer


class InferencePromptAlignedDataset(Dataset):
    """Soft-align user-only prompts exactly as inference does.

    Training datasets include the gold assistant answer. Gate diagnostics must not see
    that answer, so this wrapper removes trailing assistant messages and asks both chat
    templates to append only the assistant generation prefix.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        aligner: TokenAligner,
        max_length: int,
        soft_alignment_top_k: int,
    ) -> None:
        self.dataset = dataset
        self.aligner = aligner
        self.max_length = int(max_length)
        self.soft_alignment_top_k = max(1, int(soft_alignment_top_k))

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def inference_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        retained = [dict(message) for message in messages]
        while retained and retained[-1].get("role") == "assistant":
            retained.pop()
        if not retained or retained[-1].get("role") != "user":
            raise ValueError("Inference gate diagnostics require a final user message")
        return retained

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        messages = self.inference_messages(self.dataset[idx])
        details = self.aligner.align_chat_messages_soft(
            messages,
            add_generation_prompt=True,
            top_k=self.soft_alignment_top_k,
            return_details=True,
        )
        slm_ids = list(details["slm_ids"][: self.max_length])
        llm_ids = list(details["llm_ids"][: self.max_length])
        message_mask = torch.tensor(
            details["message_mask"][: self.max_length], dtype=torch.bool
        )
        soft = details["soft_alignment"]
        source_indices = torch.tensor(
            soft["source_indices"][: self.max_length], dtype=torch.long
        )
        source_weights = torch.tensor(
            soft["source_weights"][: self.max_length], dtype=torch.float
        )
        source_confidence = torch.tensor(
            soft.get("source_confidence", [1.0] * len(soft["source_indices"]))[
                : self.max_length
            ],
            dtype=torch.float,
        )
        source_entropy = torch.tensor(
            soft.get("source_entropy", [0.0] * len(soft["source_indices"]))[
                : self.max_length
            ],
            dtype=torch.float,
        )
        source_entropy_override = torch.tensor(
            soft.get(
                "source_entropy_override",
                [False] * len(soft["source_indices"]),
            )[: self.max_length],
            dtype=torch.bool,
        )
        source_indices, source_weights = AlignedChatDataset._renormalize_soft_alignment(
            source_indices, source_weights, len(llm_ids)
        )
        native_entropy = AlignedChatDataset._normalized_soft_alignment_entropy(
            source_weights
        )
        source_entropy = torch.where(
            source_entropy_override, source_entropy, native_entropy
        )

        kv_cache_index = torch.tensor([1, 0]).repeat(len(slm_ids), 1)
        kv_cache_index[~message_mask] = torch.tensor([-1, 0])
        return {
            "input_ids": [slm_ids, llm_ids],
            "labels": [-100] * len(slm_ids),
            "kv_cache_index": kv_cache_index,
            "messages": messages,
            "model_padding_mask": [
                torch.zeros(len(slm_ids), dtype=torch.bool),
                torch.zeros(len(llm_ids), dtype=torch.bool),
            ],
            "soft_alignment": {
                "source_indices": source_indices,
                "source_weights": source_weights,
                "source_confidence": source_confidence,
                "source_entropy": source_entropy,
                "source_entropy_override": source_entropy_override,
            },
        }


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
    low_threshold: float = 0.05,
    high_threshold: float = 0.95,
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
    stats["key_confidence_square_sum"] += float(
        (key_confidence.square() * mask_expanded).sum().item()
    )
    stats["value_confidence_square_sum"] += float(
        (value_confidence.square() * mask_expanded).sum().item()
    )
    key_selected = key_confidence.masked_select(mask_expanded.bool())
    value_selected = value_confidence.masked_select(mask_expanded.bool())
    previous_count = stats["count"] - count
    key_min = float(key_selected.min().item())
    key_max = float(key_selected.max().item())
    value_min = float(value_selected.min().item())
    value_max = float(value_selected.max().item())
    if previous_count <= 0:
        stats["key_confidence_min"] = key_min
        stats["key_confidence_max"] = key_max
        stats["value_confidence_min"] = value_min
        stats["value_confidence_max"] = value_max
    else:
        stats["key_confidence_min"] = min(stats["key_confidence_min"], key_min)
        stats["key_confidence_max"] = max(stats["key_confidence_max"], key_max)
        stats["value_confidence_min"] = min(stats["value_confidence_min"], value_min)
        stats["value_confidence_max"] = max(
            stats["value_confidence_max"], value_max
        )
    stats["key_confidence_saturation_low_count"] += float(
        (key_selected <= low_threshold).sum().item()
    )
    stats["key_confidence_saturation_high_count"] += float(
        (key_selected >= high_threshold).sum().item()
    )
    stats["value_confidence_saturation_low_count"] += float(
        (value_selected <= low_threshold).sum().item()
    )
    stats["value_confidence_saturation_high_count"] += float(
        (value_selected >= high_threshold).sum().item()
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
            "key_confidence_variance",
            "key_confidence_std",
            "key_confidence_min",
            "key_confidence_max",
            "key_confidence_saturation_low_rate",
            "key_confidence_saturation_high_rate",
            "value_confidence_variance",
            "value_confidence_std",
            "value_confidence_min",
            "value_confidence_max",
            "value_confidence_saturation_low_rate",
            "value_confidence_saturation_high_rate",
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
    key_variance = max(
        stats["key_confidence_square_sum"] / count
        - row["key_confidence_mean"] ** 2,
        0.0,
    )
    value_variance = max(
        stats["value_confidence_square_sum"] / count
        - row["value_confidence_mean"] ** 2,
        0.0,
    )
    row["key_confidence_variance"] = key_variance
    row["key_confidence_std"] = key_variance**0.5
    row["key_confidence_min"] = stats["key_confidence_min"]
    row["key_confidence_max"] = stats["key_confidence_max"]
    row["key_confidence_saturation_low_rate"] = (
        stats["key_confidence_saturation_low_count"] / count
    )
    row["key_confidence_saturation_high_rate"] = (
        stats["key_confidence_saturation_high_count"] / count
    )
    row["value_confidence_variance"] = value_variance
    row["value_confidence_std"] = value_variance**0.5
    row["value_confidence_min"] = stats["value_confidence_min"]
    row["value_confidence_max"] = stats["value_confidence_max"]
    row["value_confidence_saturation_low_rate"] = (
        stats["value_confidence_saturation_low_count"] / count
    )
    row["value_confidence_saturation_high_rate"] = (
        stats["value_confidence_saturation_high_count"] / count
    )
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
    soft_alignment = []
    for section in batch["soft_alignment"]:
        moved = {
            "source_indices": section["source_indices"].to(device),
            "source_weights": section["source_weights"].to(device),
            "source_confidence": section["source_confidence"].to(device),
        }
        for optional_key in ("source_entropy", "source_entropy_override"):
            if optional_key in section:
                moved[optional_key] = section[optional_key].to(device)
        soft_alignment.append(moved)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": batch["position_ids"].to(device),
        "labels": batch["labels"].to(device),
        "kv_cache_index": [section.to(device) for section in batch["kv_cache_index"]],
        "soft_alignment": soft_alignment,
    }


def _clear_projector_records(projectors: Iterable[Any]) -> None:
    clear_projector_gate_diagnostic_records(projectors)


def _iter_projector_records(
    projectors: Iterable[Any], target_layer_by_projector: List[int]
):
    for projector_idx, projector in enumerate(projectors):
        layer_idx = (
            target_layer_by_projector[projector_idx]
            if projector_idx < len(target_layer_by_projector)
            else projector_idx
        )
        for record in getattr(projector, "alignment_diagnostic_records", []):
            yield layer_idx, record


def _flatten_gate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        flattened = {
            key: value for key, value in row.items() if key not in {"key", "value"}
        }
        for kv in ("key", "value"):
            for name, value in row[kv].items():
                flattened[f"{kv}_confidence_{name}"] = value
        output.append(flattened)
    return output


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
    parser.add_argument("--saturation-low-threshold", type=float, default=0.05)
    parser.add_argument("--saturation-high-threshold", type=float, default=0.95)
    parser.add_argument("--relative-token-bins", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        default=(
            "local/final_results/route1_alignment_v25/diagnostics/"
            "v22_token_mlp_gate_mmlu_aux64"
        ),
    )
    args = parser.parse_args()
    if args.batch_size != 1:
        raise ValueError(
            "Gate diagnostics require --batch-size 1 because projector records "
            "do not carry a padding-validity mask."
        )

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
    llm_tokenizer = _load_teacher_tokenizer(rosetta_config["teacher_model"])

    projector_info = configure_projector_gate_diagnostics(model, enabled=True)
    gate_accumulator = GateDiagnosticsAccumulator(
        low_threshold=args.saturation_low_threshold,
        high_threshold=args.saturation_high_threshold,
        relative_token_bins=args.relative_token_bins,
    )
    gate_accumulator.note_projectors(
        projector_info["projector_count"],
        projector_info["gate_projector_count"],
        projector_info["target_layer_by_projector"],
    )

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
        soft_alignment_confidence_control_mode=rosetta_config.get(
            "soft_alignment_confidence_control_mode", "native"
        ),
        soft_alignment_confidence_constant_value=rosetta_config.get(
            "soft_alignment_confidence_constant_value"
        ),
        soft_alignment_confidence_shuffle_seed=rosetta_config.get(
            "soft_alignment_confidence_shuffle_seed", 0
        ),
    )

    raw_dataset = create_dataset(
        args.dataset_type,
        split=args.dataset_split,
        num_samples=args.num_samples,
        max_word_count=None,
    )
    aligned_dataset = InferencePromptAlignedDataset(
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
            try:
                model_inputs = _move_batch_to_device(batch, device)
                model_inputs.pop("labels", None)
                outputs = model.forward(**model_inputs, use_cache=True)
                output_loss = getattr(outputs, "loss", None)
                if output_loss is not None:
                    losses.append(float(output_loss.detach().float().cpu().item()))

                for layer_idx, record in _iter_projector_records(
                    model.projector_list,
                    projector_info["target_layer_by_projector"],
                ):
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
                        args.saturation_low_threshold,
                        args.saturation_high_threshold,
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
                            args.saturation_low_threshold,
                            args.saturation_high_threshold,
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
                            args.saturation_low_threshold,
                            args.saturation_high_threshold,
                        )
            finally:
                gate_accumulator.consume_projectors(model.projector_list)

    gate_artifact = gate_accumulator.finalize(
        {
            "eval_config": args.eval_config,
            "checkpoint_dir": rosetta_config["checkpoints_dir"],
            "dataset_type": args.dataset_type,
            "dataset_split": args.dataset_split,
            "requested_samples": args.num_samples,
            "processed_samples": batches,
            "batch_size": args.batch_size,
            "diagnostic_scope": "inference_prompt_only",
            "assistant_answer_included": False,
            "add_generation_prompt": True,
            "sample_selection": (
                "first_n_in_dataset_order; truncate aligned prompt to max_length"
            ),
        }
    )
    configure_projector_gate_diagnostics(model, enabled=False)
    if projector_info["gate_projector_count"] > 0 and gate_artifact["status"] != "ok":
        raise RuntimeError("Token/head gate projectors produced no diagnostic records")
    if len(gate_artifact["by_layer"]) != projector_info["gate_projector_count"]:
        raise RuntimeError(
            "Gate diagnostics did not cover every token/head gate projector: "
            f"observed_layers={len(gate_artifact['by_layer'])}, "
            f"expected_projectors={projector_info['gate_projector_count']}"
        )

    layer_head_rows = [
        _finalize_stats({"layer": layer, "head": head}, stats)
        for (layer, head), stats in sorted(layer_head_stats.items())
    ]
    layer_head_gate_rows = {
        (row["layer"], row["head"]): row
        for row in _flatten_gate_rows(gate_artifact["by_layer_head"])
    }
    for row in layer_head_rows:
        gate_row = layer_head_gate_rows.get((row["layer"], row["head"]), {})
        row["stage"] = gate_row.get("stage")
    layer_bucket_rows = [
        _finalize_stats({"layer": layer, "bucket": bucket}, stats)
        for (layer, bucket), stats in sorted(layer_bucket_stats.items())
    ]
    layer_stage_rows = _flatten_gate_rows(gate_artifact["by_stage"])
    token_rows = _flatten_gate_rows(gate_artifact["by_relative_token_bin"])
    summary = _finalize_stats({"scope": "global"}, global_stats)
    summary.update(
        {
            "schema_version": 2,
            "eval_config": args.eval_config,
            "checkpoint_dir": rosetta_config["checkpoints_dir"],
            "dataset_type": args.dataset_type,
            "dataset_split": args.dataset_split,
            "requested_samples": args.num_samples,
            "processed_samples": batches,
            "batch_size": args.batch_size,
            "batches": batches,
            "gate_definition": GATE_DEFINITION,
            "diagnostic_scope": "inference_prompt_only",
            "assistant_answer_included": False,
            "add_generation_prompt": True,
            "sample_selection": (
                "first_n_in_dataset_order; truncate aligned prompt to max_length"
            ),
            "variance_ddof": 0,
            "saturation_thresholds": gate_artifact["saturation_thresholds"],
            "layer_axis": gate_artifact["layer_axis"],
            "layer_stage_ranges": gate_artifact["layer_groups"],
            "dimensions": gate_artifact["dimensions"],
            "global_gate": gate_artifact["global"],
            "artifacts": {
                "gate_diagnostics": "gate_diagnostics.json",
                "layer_head": "layer_head_stats.csv",
                "layer_bucket": "layer_bucket_stats.csv",
                "layer_stage": "layer_stage_stats.csv",
                "token": "token_stats.csv",
            },
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
    _write_csv(output_dir / "layer_stage_stats.csv", layer_stage_rows)
    _write_csv(output_dir / "token_stats.csv", token_rows)
    with (output_dir / "gate_diagnostics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(gate_artifact, handle, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# Route-1 V2.2 Confidence Gate Diagnostics\n\n"
            f"- Eval config: `{args.eval_config}`\n"
            f"- Samples: `{args.num_samples}` from `{args.dataset_type}` "
            f"`{args.dataset_split}`\n"
            f"- Scope: inference prompt only; gold assistant answer removed; "
            f"generation prefix enabled\n"
            f"- Mean loss: `{summary['mean_loss']}`\n"
            f"- Global key delta abs mean: `{summary['key_delta_abs_mean']}`\n"
            f"- Global value delta abs mean: `{summary['value_delta_abs_mean']}`\n"
            f"- Layer/head stats: `layer_head_stats.csv`\n"
            f"- Layer/bucket stats: `layer_bucket_stats.csv`\n"
            f"- Early/middle/late stats: `layer_stage_stats.csv`\n"
            f"- Section-relative token-bin stats: `token_stats.csv`\n"
            f"- Full compact gate artifact: `gate_diagnostics.json`\n"
            f"- Machine summary: `summary.json`\n"
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
