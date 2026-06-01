from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class AnswerPriorRegularizationOutput:
    loss: Optional[torch.Tensor]
    metrics: Dict[str, float]


def resolve_option_token_ids(
    tokenizer: Any,
    option_labels: Sequence[str] = ("A", "B", "C", "D"),
    option_token_texts: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[int]:
    """Resolve one token id per multiple-choice answer label.

    The MMLU-style assistant target is "The correct answer is X.", so tokenizers
    such as Qwen usually emit a leading-space answer token (for example " A").
    Prefer that form, but fall back to the bare label or the final token of the
    preferred candidate for tokenizers that split whitespace separately.
    """

    option_token_ids: List[int] = []
    for label in option_labels:
        candidates = list(
            option_token_texts.get(label, []) if option_token_texts is not None else []
        )
        candidates.extend([f" {label}", label])

        fallback_id: Optional[int] = None
        for candidate in candidates:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if not ids:
                continue
            if fallback_id is None:
                fallback_id = int(ids[-1])
            if len(ids) == 1:
                option_token_ids.append(int(ids[0]))
                break
        else:
            if fallback_id is None:
                raise ValueError(f"Could not tokenize option label {label!r}.")
            option_token_ids.append(fallback_id)

    return option_token_ids


def _target_distribution(
    gold_indices: torch.Tensor,
    num_options: int,
    target: str,
    label_smoothing: float,
) -> torch.Tensor:
    one_hot = F.one_hot(gold_indices, num_classes=num_options).to(dtype=torch.float32)
    uniform = torch.full_like(one_hot, 1.0 / num_options)

    if target == "gold":
        return one_hot
    if target == "uniform":
        return uniform
    if target == "smoothed_gold":
        smoothing = min(max(float(label_smoothing), 0.0), 1.0)
        return one_hot * (1.0 - smoothing) + uniform * smoothing

    raise ValueError(
        "Unsupported answer prior target "
        f"{target!r}; expected 'gold', 'uniform', or 'smoothed_gold'."
    )


def compute_answer_prior_regularization(
    logits: torch.Tensor,
    labels: torch.Tensor,
    option_token_ids: Sequence[int],
    weight: float,
    target: str = "smoothed_gold",
    label_smoothing: float = 0.7,
    temperature: float = 1.0,
    min_positions: int = 1,
) -> AnswerPriorRegularizationOutput:
    """Compute answer-choice prior regularization at supervised ABCD positions.

    Uses causal-LM shifting: logits at position t predict the label at t + 1.
    Only positions whose shifted label is one of the configured answer option
    token ids contribute to this auxiliary loss.
    """

    metrics: Dict[str, float] = {}
    if weight <= 0.0 or not option_token_ids:
        return AnswerPriorRegularizationOutput(loss=None, metrics=metrics)
    if logits.dim() != 3 or labels.dim() != 2:
        raise ValueError("Expected logits [batch, seq, vocab] and labels [batch, seq].")
    if logits.size(0) != labels.size(0):
        raise ValueError("Logits and labels batch size must match.")

    if labels.size(1) != logits.size(1):
        # Rosetta returns logits for the final section only. Callers should pass
        # the matching labels, but this keeps the helper robust for older paths.
        labels = labels[:, -logits.size(1) :]

    if logits.size(1) < 2:
        metrics["answer_prior/positions"] = 0.0
        return AnswerPriorRegularizationOutput(loss=None, metrics=metrics)

    option_ids = torch.tensor(
        [int(token_id) for token_id in option_token_ids],
        dtype=torch.long,
        device=labels.device,
    )
    shifted_labels = labels[:, 1:]
    label_matches = shifted_labels.unsqueeze(-1).eq(option_ids)
    option_mask = label_matches.any(dim=-1)
    num_positions = int(option_mask.sum().detach().cpu().item())
    metrics["answer_prior/positions"] = float(num_positions)
    if num_positions < int(min_positions):
        return AnswerPriorRegularizationOutput(loss=None, metrics=metrics)

    shifted_logits = logits[:, :-1, :].float()
    option_logits = shifted_logits.index_select(dim=-1, index=option_ids)
    selected_logits = option_logits[option_mask]
    gold_indices = label_matches[option_mask].float().argmax(dim=-1)

    temp = max(float(temperature), 1e-6)
    log_probs = F.log_softmax(selected_logits / temp, dim=-1)
    probs = log_probs.exp()
    target_probs = _target_distribution(
        gold_indices=gold_indices,
        num_options=len(option_token_ids),
        target=target,
        label_smoothing=label_smoothing,
    ).to(device=log_probs.device)

    raw_loss = F.kl_div(log_probs, target_probs, reduction="batchmean")
    weighted_loss = raw_loss * float(weight)

    pred_mean = probs.detach().mean(dim=0)
    target_mean = target_probs.detach().mean(dim=0)
    gold_mean = F.one_hot(gold_indices, num_classes=len(option_token_ids)).float()
    gold_mean = gold_mean.detach().mean(dim=0)
    labels_for_metrics = ("A", "B", "C", "D")
    for idx, label in enumerate(labels_for_metrics[: len(option_token_ids)]):
        metrics[f"answer_prior/pred_{label}"] = float(pred_mean[idx].cpu().item())
        metrics[f"answer_prior/target_{label}"] = float(target_mean[idx].cpu().item())
        metrics[f"answer_prior/gold_{label}"] = float(gold_mean[idx].cpu().item())

    metrics["answer_prior/raw_loss"] = float(raw_loss.detach().cpu().item())
    metrics["answer_prior/weighted_loss"] = float(weighted_loss.detach().cpu().item())

    return AnswerPriorRegularizationOutput(loss=weighted_loss, metrics=metrics)
