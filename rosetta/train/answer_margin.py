from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class AnswerMarginRoutingOutput:
    loss: Optional[torch.Tensor]
    metrics: Dict[str, float]


def compute_answer_margin_routing_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    option_token_ids: Sequence[int],
    weight: float,
    objective: str = "hinge",
    margin: float = 0.5,
    temperature: float = 1.0,
    ce_weight: float = 1.0,
    hinge_weight: float = 1.0,
    min_positions: int = 1,
) -> AnswerMarginRoutingOutput:
    """Compute an option-level auxiliary loss at supervised ABCD positions.

    The regular LM loss optimizes the gold answer token against the full
    vocabulary. This helper narrows the auxiliary signal to the benchmark
    decision surface: the relative score between the gold option and the other
    answer options. It uses causal-LM shifting, so logits at position t predict
    the label at t + 1.
    """

    metrics: Dict[str, float] = {}
    if weight <= 0.0 or not option_token_ids:
        return AnswerMarginRoutingOutput(loss=None, metrics=metrics)
    if logits.dim() != 3 or labels.dim() != 2:
        raise ValueError("Expected logits [batch, seq, vocab] and labels [batch, seq].")
    if logits.size(0) != labels.size(0):
        raise ValueError("Logits and labels batch size must match.")

    if labels.size(1) != logits.size(1):
        labels = labels[:, -logits.size(1) :]

    if logits.size(1) < 2:
        metrics["answer_margin/positions"] = 0.0
        return AnswerMarginRoutingOutput(loss=None, metrics=metrics)

    option_ids = torch.tensor(
        [int(token_id) for token_id in option_token_ids],
        dtype=torch.long,
        device=labels.device,
    )
    shifted_labels = labels[:, 1:]
    label_matches = shifted_labels.unsqueeze(-1).eq(option_ids)
    option_mask = label_matches.any(dim=-1)
    num_positions = int(option_mask.sum().detach().cpu().item())
    metrics["answer_margin/positions"] = float(num_positions)
    if num_positions < int(min_positions):
        return AnswerMarginRoutingOutput(loss=None, metrics=metrics)

    temp = max(float(temperature), 1e-6)
    shifted_logits = logits[:, :-1, :].float()
    option_logits = shifted_logits.index_select(dim=-1, index=option_ids) / temp
    selected_logits = option_logits[option_mask]
    gold_indices = label_matches[option_mask].to(dtype=torch.long).argmax(dim=-1)

    gold_logits = selected_logits.gather(1, gold_indices.unsqueeze(1)).squeeze(1)
    wrong_mask = torch.ones_like(selected_logits, dtype=torch.bool)
    wrong_mask.scatter_(1, gold_indices.unsqueeze(1), False)
    max_wrong_logits = selected_logits.masked_fill(~wrong_mask, -torch.inf).max(dim=1)
    option_margin = gold_logits - max_wrong_logits.values

    ce_loss = F.cross_entropy(selected_logits, gold_indices)
    hinge_loss = F.relu(float(margin) - option_margin).mean()
    if objective == "ce":
        raw_loss = ce_loss
    elif objective == "hinge":
        raw_loss = hinge_loss
    elif objective == "ce_hinge":
        raw_loss = float(ce_weight) * ce_loss + float(hinge_weight) * hinge_loss
    else:
        raise ValueError(
            "Unsupported answer margin objective "
            f"{objective!r}; expected 'ce', 'hinge', or 'ce_hinge'."
        )

    weighted_loss = raw_loss * float(weight)

    with torch.no_grad():
        probs = F.softmax(selected_logits, dim=-1)
        pred_indices = selected_logits.argmax(dim=-1)
        pred_mean = F.one_hot(pred_indices, num_classes=len(option_token_ids)).float()
        gold_mean = F.one_hot(gold_indices, num_classes=len(option_token_ids)).float()
        labels_for_metrics = ("A", "B", "C", "D")
        for idx, label in enumerate(labels_for_metrics[: len(option_token_ids)]):
            metrics[f"answer_margin/prob_{label}"] = float(
                probs[:, idx].mean().cpu().item()
            )
            metrics[f"answer_margin/pred_{label}"] = float(
                pred_mean[:, idx].mean().cpu().item()
            )
            metrics[f"answer_margin/gold_{label}"] = float(
                gold_mean[:, idx].mean().cpu().item()
            )

        metrics["answer_margin/option_accuracy"] = float(
            pred_indices.eq(gold_indices).float().mean().cpu().item()
        )
        metrics["answer_margin/margin_mean"] = float(option_margin.mean().cpu().item())
        metrics["answer_margin/margin_min"] = float(option_margin.min().cpu().item())
        metrics["answer_margin/ce_loss"] = float(ce_loss.detach().cpu().item())
        metrics["answer_margin/hinge_loss"] = float(hinge_loss.detach().cpu().item())
        metrics["answer_margin/raw_loss"] = float(raw_loss.detach().cpu().item())
        metrics["answer_margin/weighted_loss"] = float(
            weighted_loss.detach().cpu().item()
        )

    return AnswerMarginRoutingOutput(loss=weighted_loss, metrics=metrics)
