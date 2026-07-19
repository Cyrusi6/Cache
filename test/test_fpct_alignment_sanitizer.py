from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pytest

from rosetta.model.aligner import AlignmentStrategy, TokenAligner


@dataclass
class Encoding:
    input_ids: list[int]
    offset_mapping: list[tuple[int, int]]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str) -> Any:
        return getattr(self, key)


class Backend:
    def to_str(self) -> str:
        return json.dumps(
            {
                "normalizer": {"type": "NFC"},
                "pre_tokenizer": {"type": "ByteLevel"},
                "decoder": {"type": "ByteLevel"},
                "post_processor": None,
                "model": {"type": "synthetic"},
                "added_tokens": [],
            }
        )


class IdentityTokenizer:
    def __init__(
        self,
        directory: Path,
        *,
        duplicate_offsets: bool = False,
        render_suffix: str = "",
        id_delta: int = 0,
        offset_delta: int = 0,
    ) -> None:
        self.name_or_path = str(directory)
        self.duplicate_offsets = duplicate_offsets
        self.render_suffix = render_suffix
        self.id_delta = id_delta
        self.offset_delta = offset_delta
        self.backend_tokenizer = Backend()
        self.is_fast = True
        self.chat_template = "<user>{{ content }}</user>"
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.unk_token = "<unk>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self.all_special_ids = [0, 1, 2, 3]
        self.all_special_tokens = ["<pad>", "<eos>", "<bos>", "<unk>"]
        self.special_tokens_map_extended = {
            "pad_token": "<pad>",
            "eos_token": "<eos>",
            "bos_token": "<bos>",
            "unk_token": "<unk>",
        }

    def get_vocab(self) -> dict[str, int]:
        return {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ):
        text = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        text += self.render_suffix
        if tokenize:
            return self(text, add_special_tokens=False)["input_ids"]
        return text

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **_kwargs,
    ) -> Encoding:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        for index, char in enumerate(text):
            if self.duplicate_offsets and char == "Ω":
                ids.extend((7000, 7001))
                offsets.extend(((index, index + 1), (index, index + 1)))
            else:
                ids.append(100 + ord(char) + self.id_delta)
                offsets.append(
                    (index + self.offset_delta, index + 1 + self.offset_delta)
                )
        return Encoding(ids, offsets)


def tokenizer_dirs(tmp_path: Path) -> tuple[Path, Path]:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for directory in (left, right):
        directory.mkdir()
        (directory / "tokenizer.json").write_text('{"version":"1"}\n')
        (directory / "tokenizer_config.json").write_text(
            '{"chat_template":"same"}\n'
        )
    return left, right


@pytest.mark.parametrize("duplicate_offsets", [False, True])
def test_production_exact_identity_is_hard_k1(
    tmp_path: Path, duplicate_offsets: bool
) -> None:
    left, right = tokenizer_dirs(tmp_path)
    receiver = IdentityTokenizer(left, duplicate_offsets=duplicate_offsets)
    sender = IdentityTokenizer(right, duplicate_offsets=duplicate_offsets)
    aligner = TokenAligner(receiver, sender, strategy=AlignmentStrategy.EXACT_IDENTITY)
    details = aligner.align_chat_messages_soft(
        [{"role": "user", "content": "A Ω"}],
        add_generation_prompt=True,
        top_k=4,
        return_details=True,
        apply_confidence_control=False,
    )
    for parent, eligible in enumerate(details["message_mask"]):
        row_indices = details["soft_alignment"]["source_indices"][parent]
        row_weights = details["soft_alignment"]["source_weights"][parent]
        if eligible:
            assert row_indices == [parent, -1, -1, -1]
            assert row_weights == [1.0, 0.0, 0.0, 0.0]
            assert not details["soft_alignment"]["fallback_mask"][parent]
    assert details["soft_alignment"]["fpct_extra_slots"] == 0


def test_exact_identity_render_mismatch_hard_errors(tmp_path: Path) -> None:
    left, right = tokenizer_dirs(tmp_path)
    receiver = IdentityTokenizer(left)
    sender = IdentityTokenizer(right, render_suffix="x")
    aligner = TokenAligner(receiver, sender, strategy="exact_identity")
    with pytest.raises(ValueError, match="rendered chat text mismatch"):
        aligner.align_chat_messages_soft(
            [{"role": "user", "content": "same"}], return_details=True
        )


@pytest.mark.parametrize(
    "sender_kwargs,error",
    [
        ({"id_delta": 1}, "input IDs mismatch"),
        ({"offset_delta": 1}, "offset mappings mismatch"),
    ],
)
def test_exact_identity_runtime_mismatch_hard_errors(
    tmp_path: Path, sender_kwargs: dict[str, int], error: str
) -> None:
    left, right = tokenizer_dirs(tmp_path)
    receiver = IdentityTokenizer(left)
    sender = IdentityTokenizer(right, **sender_kwargs)
    aligner = TokenAligner(receiver, sender, strategy="exact_identity")
    with pytest.raises(ValueError, match=error):
        aligner.align_chat_messages_soft(
            [{"role": "user", "content": "same"}], return_details=True
        )


def details_for_sanitizer(
    *,
    receiver_offsets=((0, 2),),
    source_offsets=((0, 1), (1, 2)),
    rows=((0, 1, -1, -1),),
    weights=((0.5, 0.5, 0.0, 0.0),),
    positive_counts=(2,),
) -> dict[str, Any]:
    return {
        "slm_ids": list(range(len(receiver_offsets))),
        "llm_ids": list(range(len(source_offsets))),
        "slm_offsets": list(receiver_offsets),
        "llm_offsets": list(source_offsets),
        "content_spans_slm": [(0, 2)],
        "content_spans_llm": [(0, 2)],
        "sections": [
            {
                "type": "message",
                "slm_range": (0, len(receiver_offsets)),
                "llm_range": (0, len(source_offsets)),
            }
        ],
        "message_mask": [True] * len(receiver_offsets),
        "soft_alignment": {
            "source_indices": [list(row) for row in rows],
            "source_weights": [list(row) for row in weights],
            "source_entropy": [1.0] * len(rows),
            "source_entropy_override": [False] * len(rows),
            "positive_overlap_counts": list(positive_counts),
        },
    }


def test_genuine_partition_is_preserved_and_certified() -> None:
    result = TokenAligner.sanitize_fpct_soft_alignment(details_for_sanitizer())
    soft = result["soft_alignment"]
    assert soft["source_indices"][0] == [0, 1, -1, -1]
    assert soft["source_weights"][0] == [0.5, 0.5, 0.0, 0.0]
    assert soft["fpct_certified_mask"] == [True]
    assert soft["fpct_offset_uncertified_mask"] == [False]


def test_overlapping_receiver_rows_are_common_slot0_one_hot() -> None:
    details = details_for_sanitizer(
        receiver_offsets=((0, 2), (1, 2)),
        rows=((0, 1, -1, -1), (0, 1, -1, -1)),
        weights=((0.5, 0.5, 0.0, 0.0), (0.5, 0.5, 0.0, 0.0)),
        positive_counts=(2, 2),
    )
    result = TokenAligner.sanitize_fpct_soft_alignment(details)
    soft = result["soft_alignment"]
    assert soft["source_indices"] == [[0, -1, -1, -1], [0, -1, -1, -1]]
    assert soft["source_weights"] == [[1.0, 0.0, 0.0, 0.0]] * 2
    assert soft["fpct_offset_uncertified_mask"] == [True, True]
    assert soft["source_entropy"] == [0.0, 0.0]


def test_duplicate_source_offsets_and_incomplete_topk_are_demoted() -> None:
    duplicate = details_for_sanitizer(source_offsets=((0, 2), (0, 2)))
    duplicate_soft = TokenAligner.sanitize_fpct_soft_alignment(duplicate)[
        "soft_alignment"
    ]
    assert duplicate_soft["source_weights"][0] == [1.0, 0.0, 0.0, 0.0]
    assert duplicate_soft["fpct_offset_uncertified_mask"] == [True]

    incomplete = details_for_sanitizer(
        source_offsets=((0, 1), (1, 2), (0, 2)), positive_counts=(3,)
    )
    incomplete_soft = TokenAligner.sanitize_fpct_soft_alignment(incomplete)[
        "soft_alignment"
    ]
    assert incomplete_soft["source_weights"][0] == [1.0, 0.0, 0.0, 0.0]
    assert incomplete_soft["fpct_offset_uncertified_mask"] == [True]


def test_source_truncation_revalidates_support() -> None:
    result = TokenAligner.sanitize_fpct_soft_alignment(
        details_for_sanitizer(), source_length=1
    )
    soft = result["soft_alignment"]
    assert soft["source_indices"][0] == [0, -1, -1, -1]
    assert soft["source_weights"][0] == [1.0, 0.0, 0.0, 0.0]
    assert soft["fpct_sanitized_candidate_count"] == [1]
    assert soft["fpct_certified_mask"] == [False]


def test_source_truncation_recomputes_retained_overlap_exhaustiveness() -> None:
    result = TokenAligner.sanitize_fpct_soft_alignment(
        details_for_sanitizer(
            source_offsets=((0, 1), (1, 2), (0, 2)), positive_counts=(3,)
        ),
        source_length=2,
    )
    soft = result["soft_alignment"]
    assert soft["fpct_certified_mask"] == [True]
    assert soft["source_indices"][0] == [0, 1, -1, -1]
    assert soft["fpct_extra_slots"] == 1


def test_receiver_truncation_excludes_nonretained_overlap_alias() -> None:
    details = details_for_sanitizer(
        receiver_offsets=((0, 2), (1, 2)),
        rows=((0, 1, -1, -1), (0, 1, -1, -1)),
        weights=((0.5, 0.5, 0.0, 0.0), (0.5, 0.5, 0.0, 0.0)),
        positive_counts=(2, 2),
    )
    result = TokenAligner.sanitize_fpct_soft_alignment(details, target_length=1)
    assert result["soft_alignment"]["fpct_certified_mask"][0]


@pytest.mark.parametrize(
    "rows,weights,error",
    [
        (
            ((0, 1, -1, -1),),
            ((float("nan"), 0.5, 0.0, 0.0),),
            "nonfinite",
        ),
        (
            ((0, 1, -1, -1),),
            ((-0.5, 1.5, 0.0, 0.0),),
            "negative",
        ),
        (
            ((0, 9, -1, -1),),
            ((0.5, 0.5, 0.0, 0.0),),
            "invalid source",
        ),
    ],
)
def test_sanitizer_integrity_anomalies_hard_error(rows, weights, error) -> None:
    with pytest.raises(ValueError, match=error):
        TokenAligner.sanitize_fpct_soft_alignment(
            details_for_sanitizer(rows=rows, weights=weights)
        )


def test_uncertified_row_requires_raw_slot0_anchor() -> None:
    details = details_for_sanitizer(
        receiver_offsets=((0, 2), (1, 2)),
        rows=((-1, 0, 1, -1), (-1, 0, 1, -1)),
        weights=((0.0, 0.5, 0.5, 0.0), (0.0, 0.5, 0.5, 0.0)),
        positive_counts=(2, 2),
    )
    with pytest.raises(ValueError, match="slot-0 anchor"):
        TokenAligner.sanitize_fpct_soft_alignment(details)
