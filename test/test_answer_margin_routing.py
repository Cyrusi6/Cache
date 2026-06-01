from __future__ import annotations

import torch

from rosetta.train.answer_margin import compute_answer_margin_routing_loss


def test_answer_margin_routing_uses_shifted_answer_position() -> None:
    option_ids = [10, 11, 12, 13]
    labels = torch.tensor([[100, 101, 11, 102]])
    logits = torch.zeros(1, 4, 32)
    logits[0, 1, 11] = 5.0
    logits[0, 1, 10] = 2.0

    good = compute_answer_margin_routing_loss(
        logits=logits,
        labels=labels,
        option_token_ids=option_ids,
        weight=0.1,
        objective="hinge",
        margin=0.5,
    )

    bad_logits = logits.clone()
    bad_logits[0, 1, 11] = 1.0
    bad_logits[0, 1, 10] = 5.0
    bad = compute_answer_margin_routing_loss(
        logits=bad_logits,
        labels=labels,
        option_token_ids=option_ids,
        weight=0.1,
        objective="hinge",
        margin=0.5,
    )

    assert good.loss is not None
    assert bad.loss is not None
    assert good.metrics["answer_margin/positions"] == 1.0
    assert good.metrics["answer_margin/gold_B"] == 1.0
    assert good.metrics["answer_margin/option_accuracy"] == 1.0
    assert good.loss.item() < bad.loss.item()


def test_answer_margin_routing_ce_objective_penalizes_wrong_option() -> None:
    option_ids = [10, 11, 12, 13]
    labels = torch.tensor([[100, 101, 12, 102]])
    logits = torch.zeros(1, 4, 32)
    logits[0, 1, 12] = 4.0

    good = compute_answer_margin_routing_loss(
        logits=logits,
        labels=labels,
        option_token_ids=option_ids,
        weight=0.1,
        objective="ce",
    )

    bad_logits = logits.clone()
    bad_logits[0, 1, 12] = 0.0
    bad_logits[0, 1, 13] = 4.0
    bad = compute_answer_margin_routing_loss(
        logits=bad_logits,
        labels=labels,
        option_token_ids=option_ids,
        weight=0.1,
        objective="ce",
    )

    assert good.loss is not None
    assert bad.loss is not None
    assert good.loss.item() < bad.loss.item()


def test_answer_margin_routing_ignores_batches_without_answer_token() -> None:
    labels = torch.tensor([[100, 101, 102]])
    logits = torch.zeros(1, 3, 32)

    output = compute_answer_margin_routing_loss(
        logits=logits,
        labels=labels,
        option_token_ids=[10, 11, 12, 13],
        weight=0.1,
    )

    assert output.loss is None
    assert output.metrics["answer_margin/positions"] == 0.0
