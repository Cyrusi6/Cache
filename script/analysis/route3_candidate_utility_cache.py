"""
Build an offline Route-3 forced-candidate utility cache.

The cache stores per-sample candidate targets/utilities so Route-3 training can
consume stable utility labels without replaying every candidate inside each
training step.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.train.SFT_train import (
    _candidate_replay_target_from_forced_ranks,
    _set_learned_alignment_replay_target,
    _set_learned_alignment_replay_utility,
    _set_learned_alignment_replay_utility_valid,
    detect_training_mode,
    enable_full_determinism,
    load_config,
    resolve_option_token_ids,
    set_seed,
    setup_models,
)
from rosetta.train.dataset_adapters import (
    AlignedChatDataset,
    ChatDataset,
    RosettaDataCollator,
    create_dataset,
)


def _to_device_soft_alignment(
    soft_alignment: Optional[List[Dict[str, torch.Tensor]]],
    device: str,
) -> Optional[List[Dict[str, torch.Tensor]]]:
    if soft_alignment is None:
        return None
    return [
        {
            "source_indices": section["source_indices"].to(device),
            "source_weights": section["source_weights"].to(device),
            "source_confidence": section.get(
                "source_confidence",
                torch.ones_like(
                    section["source_indices"][..., 0],
                    dtype=torch.float,
                ),
            ).to(device),
        }
        for section in soft_alignment
    ]


def _target_metrics(target: torch.Tensor, utility: torch.Tensor) -> Dict[str, float]:
    target = target.detach().float().cpu()
    utility = utility.detach().float().cpu()
    top2 = torch.topk(target, k=min(2, target.numel())).values
    target_margin = float((top2[0] - top2[-1]).item()) if top2.numel() > 1 else 0.0
    entropy = -float((target * target.clamp_min(1e-8).log()).sum().item())
    denom = math.log(max(2, target.numel()))
    utility_top2 = torch.topk(utility, k=min(2, utility.numel())).values
    utility_margin = (
        float((utility_top2[0] - utility_top2[-1]).item())
        if utility_top2.numel() > 1
        else 0.0
    )
    return {
        "target_anchor": float(target[0].item()) if target.numel() > 0 else 0.0,
        "target_top1": float(target.max().item()) if target.numel() > 0 else 0.0,
        "target_entropy": entropy / denom,
        "target_margin": target_margin,
        "best_rank": float(target.argmax().item()) if target.numel() > 0 else -1.0,
        "utility_margin": utility_margin,
    }


def _build_dataset(
    cfg: Dict[str, Any],
    aligner: Any,
    main_tokenizer: Any,
    llm_tokenizer: Any,
) -> Any:
    model_config = cfg["model"]
    training_config = cfg["training"]
    data_config = cfg["data"]
    instruct_ds = create_dataset(
        dataset_type=data_config["type"],
        **data_config["kwargs"],
    )
    if model_config.get("is_do_alignment", False) and aligner is not None:
        return AlignedChatDataset(
            instruct_ds,
            aligner,
            max_length=training_config.get("max_length", 2048),
            soft_alignment_top_k=model_config.get("soft_alignment_top_k", 4),
        )
    return ChatDataset(instruct_ds, main_tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Route-3 offline candidate utility cache."
    )
    parser.add_argument("--config", required=True, help="Training config to mirror.")
    parser.add_argument("--output", required=True, help="JSONL cache output path.")
    parser.add_argument("--summary", default=None, help="Optional summary JSON path.")
    parser.add_argument("--limit", type=int, default=None, help="Max samples.")
    parser.add_argument(
        "--split",
        choices=["train", "eval", "all"],
        default="train",
        help="Subset after applying the config train/eval split.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    enable_full_determinism()

    training_mode = detect_training_mode(cfg["model"])
    if training_mode != "rosetta":
        raise ValueError("Route-3 candidate utility cache requires rosetta mode.")

    model, main_tokenizer, aligner, llm_tokenizer = setup_models(
        cfg["model"],
        training_mode,
        args.device,
        torch.bfloat16,
    )
    model = model.to(args.device).eval()

    raw_candidate_replay_config = cfg["training"].get("candidate_replay_alignment")
    if raw_candidate_replay_config is None:
        raise ValueError("training.candidate_replay_alignment is required.")
    candidate_replay_config = dict(raw_candidate_replay_config)
    if not candidate_replay_config or not candidate_replay_config.get("enabled", False):
        raise ValueError("training.candidate_replay_alignment.enabled must be true.")
    score_mode = str(candidate_replay_config.get("score_mode", "task_loss"))
    if (
        score_mode in {"answer_token_ce", "answer_suffix_ce", "answer_margin"}
        and "option_token_ids" not in candidate_replay_config
    ):
        candidate_replay_config["option_token_ids"] = resolve_option_token_ids(
            main_tokenizer,
            option_labels=candidate_replay_config.get(
                "option_labels",
                ["A", "B", "C", "D"],
            ),
            option_token_texts=candidate_replay_config.get("option_token_texts"),
        )

    dataset = _build_dataset(cfg, aligner, main_tokenizer, llm_tokenizer)
    all_indices = list(range(len(dataset)))
    train_size = int(cfg["data"]["train_ratio"] * len(dataset))
    generator = torch.Generator().manual_seed(cfg["training"]["seed"])
    train_subset, eval_subset = torch.utils.data.random_split(
        all_indices,
        [train_size, len(dataset) - train_size],
        generator=generator,
    )
    if args.split == "train":
        indices = list(train_subset)
    elif args.split == "eval":
        indices = list(eval_subset)
    else:
        indices = all_indices
    if args.limit is not None:
        indices = indices[: args.limit]

    collator = RosettaDataCollator(
        slm_tokenizer=main_tokenizer,
        llm_tokenizer=llm_tokenizer,
        pad_to_multiple_of=cfg["training"].get("pad_to_multiple_of", None),
        max_length=cfg["training"].get("max_length", 2048),
        aligner=aligner,
        do_alignment=cfg["model"].get("is_do_alignment", False),
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    )

    totals: Dict[str, float] = {
        "samples": 0.0,
        "cached": 0.0,
        "target_anchor": 0.0,
        "target_top1": 0.0,
        "target_entropy": 0.0,
        "target_margin": 0.0,
        "utility_margin": 0.0,
        "best_rank": 0.0,
    }
    with output_path.open("w") as f:
        for original_idx, batch in tqdm(
            zip(indices, loader),
            total=len(indices),
            desc="candidate-cache",
        ):
            input_ids = [x.to(args.device) for x in batch["input_ids"]]
            attention_mask = [x.to(args.device) for x in batch["attention_mask"]]
            position_ids = batch["position_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            kv_cache_index = [x.to(args.device) for x in batch["kv_cache_index"]]
            soft_alignment = _to_device_soft_alignment(
                batch.get("soft_alignment"),
                args.device,
            )

            _set_learned_alignment_replay_target(model, None)
            _set_learned_alignment_replay_utility(model, None)
            _set_learned_alignment_replay_utility_valid(model, None)
            target, utility, metrics = _candidate_replay_target_from_forced_ranks(
                model=model,
                kv_cache_index=kv_cache_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                soft_alignment=soft_alignment,
                config=candidate_replay_config,
            )
            totals["samples"] += 1.0
            if target is None or utility is None:
                row = {"idx": int(original_idx), "cached": False, "metrics": metrics}
            else:
                utility_valid = torch.ones_like(utility, dtype=torch.bool)
                row_metrics = _target_metrics(target, utility)
                row = {
                    "idx": int(original_idx),
                    "cached": True,
                    "target": [
                        float(x) for x in target.detach().float().cpu().tolist()
                    ],
                    "utility": [
                        float(x) for x in utility.detach().float().cpu().tolist()
                    ],
                    "utility_valid": [
                        bool(x) for x in utility_valid.detach().cpu().tolist()
                    ],
                    "metrics": {**metrics, **row_metrics},
                }
                totals["cached"] += 1.0
                for key in (
                    "target_anchor",
                    "target_top1",
                    "target_entropy",
                    "target_margin",
                    "utility_margin",
                    "best_rank",
                ):
                    totals[key] += row_metrics[key]
            f.write(json.dumps(row) + "\n")

    cached = max(totals["cached"], 1.0)
    summary = {
        "config": args.config,
        "output": str(output_path),
        "split": args.split,
        "num_samples": int(totals["samples"]),
        "num_cached": int(totals["cached"]),
        "cache_rate": totals["cached"] / max(totals["samples"], 1.0),
        "target_anchor": totals["target_anchor"] / cached,
        "target_top1": totals["target_top1"] / cached,
        "target_entropy": totals["target_entropy"] / cached,
        "target_margin": totals["target_margin"] / cached,
        "utility_margin": totals["utility_margin"] / cached,
        "best_rank": totals["best_rank"] / cached,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
