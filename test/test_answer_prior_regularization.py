from __future__ import annotations

from typing import List

import torch

from rosetta.train.answer_prior import (
    compute_answer_prior_regularization,
    resolve_option_token_ids,
)


class TinyTokenizer:
    def __init__(self) -> None:
        self.vocab = {
            "A": [1],
            "B": [2],
            "C": [3],
            "D": [4],
            " A": [10],
            " B": [11],
            " C": [12],
            " D": [13],
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self.vocab[text]


def test_resolve_option_token_ids_prefers_leading_space_tokens() -> None:
    tokenizer = TinyTokenizer()

    token_ids = resolve_option_token_ids(tokenizer)

    assert token_ids == [10, 11, 12, 13]


def test_answer_prior_regularization_uses_shifted_answer_position() -> None:
    option_ids = [10, 11, 12, 13]
    labels = torch.tensor([[100, 101, 11, 102]])
    logits = torch.zeros(1, 4, 32)
    logits[0, 1, 11] = 8.0

    good = compute_answer_prior_regularization(
        logits=logits,
        labels=labels,
        option_token_ids=option_ids,
        weight=0.1,
        target="smoothed_gold",
        label_smoothing=0.2,
    )

    bad_logits = logits.clone()
    bad_logits[0, 1, 11] = 0.0
    bad_logits[0, 1, 10] = 8.0
    bad = compute_answer_prior_regularization(
        logits=bad_logits,
        labels=labels,
        option_token_ids=option_ids,
        weight=0.1,
        target="smoothed_gold",
        label_smoothing=0.2,
    )

    assert good.loss is not None
    assert bad.loss is not None
    assert good.metrics["answer_prior/positions"] == 1.0
    assert good.metrics["answer_prior/gold_B"] == 1.0
    assert good.loss.item() < bad.loss.item()


def test_answer_prior_regularization_ignores_batches_without_answer_token() -> None:
    labels = torch.tensor([[100, 101, 102]])
    logits = torch.zeros(1, 3, 32)

    output = compute_answer_prior_regularization(
        logits=logits,
        labels=labels,
        option_token_ids=[10, 11, 12, 13],
        weight=0.1,
    )

    assert output.loss is None
    assert output.metrics["answer_prior/positions"] == 0.0
