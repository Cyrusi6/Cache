"""
Route-1 tokenizer alignment diagnostics.

Example:
    python script/analysis/route1_alignment_diagnostics.py \
      --config local/tmp/train_recipes/route1_alignment/qwen3_0.6b_tinyllama1.1b_soft_span_overlap_smoke256.json \
      --limit 512 \
      --output-dir local/final_results/route1_alignment/diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import yaml
from transformers import AutoTokenizer

from rosetta.model.aligner import AlignmentStrategy, TokenAligner
from rosetta.train.dataset_adapters import create_dataset
from rosetta.utils.evaluate import set_default_chat_template


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def entropy(weights: List[float]) -> float:
    return -sum(w * math.log(w) for w in weights if w > 0)


def summarize_sample(messages: List[Dict[str, str]], longest: TokenAligner, span: TokenAligner,
                     soft: TokenAligner, top_k: int) -> Dict[str, Any]:
    longest_details = longest.align_chat_messages(
        messages, add_generation_prompt=False, return_details=True
    )
    span_details = span.align_chat_messages(
        messages, add_generation_prompt=False, return_details=True
    )
    soft_details = soft.align_chat_messages_soft(
        messages, add_generation_prompt=False, return_details=True, top_k=top_k
    )

    hard_message_positions = [
        i for i, is_msg in enumerate(longest_details["message_mask"]) if is_msg
    ]
    comparable_positions = [
        i for i in hard_message_positions
        if i < len(span_details["llm_ids_padded"])
        and i < len(longest_details["llm_ids_padded"])
    ]
    changed = sum(
        1
        for i in comparable_positions
        if longest_details["llm_ids_padded"][i] != span_details["llm_ids_padded"][i]
    )

    soft_message_positions = [
        i for i, is_msg in enumerate(soft_details["message_mask"]) if is_msg
    ]
    soft_alignment = soft_details["soft_alignment"]
    fallback_count = 0
    no_positive_count = 0
    one_to_many_count = 0
    nonzero_candidate_total = 0
    entropy_total = 0.0
    top1_weight_total = 0.0
    top1_boundary_hit_count = 0
    confidence_total = 0.0
    confidence_gated_count = 0
    fallback_confidence_total = 0.0
    top1_counts: Dict[int, int] = {}
    top1_boundary_hits = soft_alignment.get(
        "top1_boundary_hit_mask", [False] * len(soft_alignment["source_indices"])
    )
    source_confidence = soft_alignment.get(
        "source_confidence", [1.0] * len(soft_alignment["source_indices"])
    )

    for i in soft_message_positions:
        indices = soft_alignment["source_indices"][i]
        weights = soft_alignment["source_weights"][i]
        positive_count = soft_alignment["positive_overlap_counts"][i]
        boundary_hit = bool(top1_boundary_hits[i])
        confidence = float(source_confidence[i])
        is_fallback = bool(soft_alignment["fallback_mask"][i])
        nonzero = sum(1 for weight in weights if weight > 0)

        fallback_count += int(is_fallback)
        no_positive_count += int(positive_count == 0)
        one_to_many_count += int(nonzero > 1)
        nonzero_candidate_total += nonzero
        entropy_total += entropy(weights)
        top1_weight_total += float(weights[0]) if weights else 0.0
        top1_boundary_hit_count += int(boundary_hit)
        confidence_total += confidence
        confidence_gated_count += int(confidence < 0.999)
        if is_fallback:
            fallback_confidence_total += confidence

        if indices and indices[0] >= 0 and weights and weights[0] > 0:
            top1_counts[indices[0]] = top1_counts.get(indices[0], 0) + 1

    many_to_one_sources = sum(1 for count in top1_counts.values() if count > 1)
    total_top1_sources = len(top1_counts)

    return {
        "hard_message_tokens": len(comparable_positions),
        "hard_longest_span_changed": changed,
        "soft_message_tokens": len(soft_message_positions),
        "soft_fallback_count": fallback_count,
        "soft_no_positive_count": no_positive_count,
        "soft_one_to_many_count": one_to_many_count,
        "soft_nonzero_candidate_total": nonzero_candidate_total,
        "soft_entropy_total": entropy_total,
        "soft_top1_weight_total": top1_weight_total,
        "soft_top1_boundary_hit_count": top1_boundary_hit_count,
        "soft_confidence_total": confidence_total,
        "soft_confidence_gated_count": confidence_gated_count,
        "soft_fallback_confidence_total": fallback_confidence_total,
        "soft_many_to_one_sources": many_to_one_sources,
        "soft_total_top1_sources": total_top1_sources,
        "slm_tokens": len(soft_details["slm_ids"]),
        "llm_tokens": len(soft_details["llm_ids"]),
    }


def ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose route-1 tokenizer alignment")
    parser.add_argument("--config", required=True, help="Training recipe JSON/YAML")
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--score-mode", default=None)
    parser.add_argument("--boundary-bonus", type=float, default=None)
    parser.add_argument("--boundary-tolerance", type=int, default=None)
    parser.add_argument("--min-weight", type=float, default=None)
    parser.add_argument("--confidence-mode", default=None)
    parser.add_argument("--confidence-alpha", type=float, default=None)
    parser.add_argument("--confidence-floor", type=float, default=None)
    parser.add_argument("--fallback-confidence", type=float, default=None)
    parser.add_argument("--output-dir", default="local/final_results/route1_alignment/diagnostics")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    top_k = args.top_k or int(model_cfg.get("soft_alignment_top_k", 4))
    score_mode = args.score_mode or model_cfg.get("soft_alignment_score_mode", "overlap")
    boundary_bonus = (
        args.boundary_bonus
        if args.boundary_bonus is not None
        else model_cfg.get("soft_alignment_boundary_bonus", 0.0)
    )
    boundary_tolerance = (
        args.boundary_tolerance
        if args.boundary_tolerance is not None
        else model_cfg.get("soft_alignment_boundary_tolerance", 1)
    )
    min_weight = (
        args.min_weight
        if args.min_weight is not None
        else model_cfg.get("soft_alignment_min_weight", 0.0)
    )
    confidence_mode = (
        args.confidence_mode
        if args.confidence_mode is not None
        else model_cfg.get("soft_alignment_confidence_mode", "none")
    )
    confidence_alpha = (
        args.confidence_alpha
        if args.confidence_alpha is not None
        else model_cfg.get("soft_alignment_confidence_alpha", 0.5)
    )
    confidence_floor = (
        args.confidence_floor
        if args.confidence_floor is not None
        else model_cfg.get("soft_alignment_confidence_floor", 0.0)
    )
    fallback_confidence = (
        args.fallback_confidence
        if args.fallback_confidence is not None
        else model_cfg.get("soft_alignment_fallback_confidence", 1.0)
    )

    slm_tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    llm_tokenizer = AutoTokenizer.from_pretrained(model_cfg["teacher_model"])
    if slm_tokenizer.pad_token is None:
        slm_tokenizer.pad_token = slm_tokenizer.eos_token
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    set_default_chat_template(slm_tokenizer, model_cfg["base_model"])
    set_default_chat_template(llm_tokenizer, model_cfg["teacher_model"])

    dataset = create_dataset(
        dataset_type=data_cfg["type"],
        **data_cfg["kwargs"],
    )
    limit = min(args.limit, len(dataset))

    longest = TokenAligner(slm_tokenizer, llm_tokenizer, AlignmentStrategy.LONGEST)
    span = TokenAligner(slm_tokenizer, llm_tokenizer, AlignmentStrategy.SPAN_OVERLAP)
    soft_strategy = AlignmentStrategy(
        model_cfg.get("alignment_strategy", "soft_span_overlap")
    )
    if soft_strategy not in {
        AlignmentStrategy.SOFT_SPAN_OVERLAP,
        AlignmentStrategy.SOFT_SPAN_OVERLAP_V2,
    }:
        soft_strategy = AlignmentStrategy.SOFT_SPAN_OVERLAP
    soft = TokenAligner(
        slm_tokenizer,
        llm_tokenizer,
        soft_strategy,
        soft_alignment_score_mode=score_mode,
        soft_alignment_boundary_bonus=boundary_bonus,
        soft_alignment_boundary_tolerance=boundary_tolerance,
        soft_alignment_min_weight=min_weight,
        soft_alignment_confidence_mode=confidence_mode,
        soft_alignment_confidence_alpha=confidence_alpha,
        soft_alignment_confidence_floor=confidence_floor,
        soft_alignment_fallback_confidence=fallback_confidence,
    )

    rows: List[Dict[str, Any]] = []
    totals: Dict[str, float] = {}
    for idx in range(limit):
        sample = summarize_sample(dataset[idx], longest, span, soft, top_k=top_k)
        sample["sample_idx"] = idx
        rows.append(sample)
        for key, value in sample.items():
            if key == "sample_idx":
                continue
            totals[key] = totals.get(key, 0.0) + float(value)

    summary = {
        "config": args.config,
        "base_model": model_cfg["base_model"],
        "teacher_model": model_cfg["teacher_model"],
        "limit": limit,
        "top_k": top_k,
        "soft_strategy": soft_strategy.value,
        "score_mode": score_mode,
        "boundary_bonus": boundary_bonus,
        "boundary_tolerance": boundary_tolerance,
        "min_weight": min_weight,
        "confidence_mode": confidence_mode,
        "confidence_alpha": confidence_alpha,
        "confidence_floor": confidence_floor,
        "fallback_confidence": fallback_confidence,
        "totals": totals,
        "rates": {
            "hard_longest_span_change_rate": ratio(
                totals.get("hard_longest_span_changed", 0.0),
                totals.get("hard_message_tokens", 0.0),
            ),
            "soft_fallback_rate": ratio(
                totals.get("soft_fallback_count", 0.0),
                totals.get("soft_message_tokens", 0.0),
            ),
            "soft_no_positive_rate": ratio(
                totals.get("soft_no_positive_count", 0.0),
                totals.get("soft_message_tokens", 0.0),
            ),
            "soft_one_to_many_rate": ratio(
                totals.get("soft_one_to_many_count", 0.0),
                totals.get("soft_message_tokens", 0.0),
            ),
            "soft_many_to_one_source_rate": ratio(
                totals.get("soft_many_to_one_sources", 0.0),
                totals.get("soft_total_top1_sources", 0.0),
            ),
            "soft_avg_nonzero_candidates": ratio(
                totals.get("soft_nonzero_candidate_total", 0.0),
                totals.get("soft_message_tokens", 0.0),
            ),
            "soft_avg_entropy": ratio(
                totals.get("soft_entropy_total", 0.0),
                totals.get("soft_message_tokens", 0.0),
            ),
            "soft_avg_top1_weight": ratio(
                totals.get("soft_top1_weight_total", 0.0),
                totals.get("soft_message_tokens", 0.0),
            ),
            "soft_top1_boundary_hit_rate": ratio(
                totals.get("soft_top1_boundary_hit_count", 0.0),
                totals.get("soft_message_tokens", 0.0),
            ),
            "soft_avg_source_confidence": ratio(
                totals.get("soft_confidence_total", 0.0),
                totals.get("soft_message_tokens", 0.0),
            ),
            "soft_confidence_gated_rate": ratio(
                totals.get("soft_confidence_gated_count", 0.0),
                totals.get("soft_message_tokens", 0.0),
            ),
            "soft_avg_fallback_confidence": ratio(
                totals.get("soft_fallback_confidence_total", 0.0),
                totals.get("soft_fallback_count", 0.0),
            ),
            "avg_slm_tokens": ratio(totals.get("slm_tokens", 0.0), limit),
            "avg_llm_tokens": ratio(totals.get("llm_tokens", 0.0), limit),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    confidence_tag = confidence_mode
    if confidence_mode != "none":
        confidence_tag = (
            f"{confidence_mode}_a{confidence_alpha:g}_"
            f"f{confidence_floor:g}_fb{fallback_confidence:g}"
        )
    stem = (
        f"route1_alignment_diag_{score_mode}_{confidence_tag}_"
        f"topk{top_k}_n{limit}"
    )
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["sample_idx"])
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Route-1 Alignment Diagnostics\n\n")
        for key, value in summary["rates"].items():
            f.write(f"- `{key}`: {value:.6f}\n")
        f.write("\n")
        f.write(f"JSON: `{json_path}`\n\n")
        f.write(f"CSV: `{csv_path}`\n")

    print(f"Saved diagnostics to {json_path}")
    print(json.dumps(summary["rates"], indent=2))


if __name__ == "__main__":
    main()
