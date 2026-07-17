from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rosetta.model.aligner import AlignmentStrategy, TokenAligner
from rosetta.model.projector import C2CProjector
from rosetta.model.wrapper import RosettaModel
from rosetta.train.dataset_adapters import AlignedChatDataset, RosettaDataCollator
from script.train.SFT_train import (
    _candidate_replay_answer_margin_from_logits,
    _candidate_replay_score_loss_from_logits,
    _cached_candidate_replay_target_from_batch,
    _load_projector_checkpoint_dir,
)


@dataclass
class FakeEncoding:
    input_ids: List[int]
    offset_mapping: List[Tuple[int, int]]

    def get(self, key: str):
        return getattr(self, key)

    def __getitem__(self, key: str):
        return getattr(self, key)


class FakeTokenizer:
    def __init__(self, mode: str):
        self.mode = mode
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.unk_token = "<unk>"
        self.vocab: Dict[str, int] = {
            self.pad_token: 0,
            self.eos_token: 1,
            self.bos_token: 2,
            self.unk_token: 3,
        }
        self.reverse_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self.all_special_ids = [0, 1, 2, 3]

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            token_id = len(self.vocab)
            self.vocab[token] = token_id
            self.reverse_vocab[token_id] = token
        return self.vocab[token]

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ):
        text = ""
        for message in messages:
            text += f"<{message['role']}>"
            text += message["content"]
            text += f"</{message['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return self(text, add_special_tokens=False)["input_ids"]
        return text

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **_kwargs,
    ):
        tokens: List[str] = []
        offsets: List[Tuple[int, int]] = []
        i = 0
        while i < len(text):
            if text[i] == "<":
                end = text.find(">", i)
                if end >= 0:
                    tokens.append(text[i : end + 1])
                    offsets.append((i, end + 1))
                    i = end + 1
                    continue
            if self.mode == "char":
                tokens.append(text[i : i + 1])
                offsets.append((i, i + 1))
                i += 1
                continue

            start = i
            if text[i].isspace():
                while i < len(text) and text[i].isspace():
                    i += 1
            elif text[i].isalnum():
                while i < len(text) and text[i].isalnum():
                    i += 1
            else:
                i += 1
            tokens.append(text[start:i])
            offsets.append((start, i))

        input_ids = [self._id(token) for token in tokens]
        return FakeEncoding(input_ids=input_ids, offset_mapping=offsets)

    def encode(self, text: str, add_special_tokens: bool = False, return_tensors=None):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ):
        return "".join(
            self.reverse_vocab.get(token_id, self.unk_token) for token_id in token_ids
        )

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)


def test_span_overlap_same_tokenizer_matches_message_tokens():
    tokenizer = FakeTokenizer("char")
    aligner = TokenAligner(
        tokenizer, tokenizer, strategy=AlignmentStrategy.SPAN_OVERLAP
    )

    messages = [{"role": "user", "content": "A 12, test."}]
    details = aligner.align_chat_messages(messages, return_details=True)

    assert len(details["slm_ids_padded"]) == len(details["llm_ids_padded"])
    assert details["slm_ids_padded"] == details["llm_ids_padded"]


def test_span_overlap_cross_tokenizer_outputs_equal_length_without_empty_mapping():
    slm_tokenizer = FakeTokenizer("char")
    llm_tokenizer = FakeTokenizer("word")
    aligner = TokenAligner(
        slm_tokenizer,
        llm_tokenizer,
        strategy=AlignmentStrategy.SPAN_OVERLAP,
    )

    messages = [{"role": "user", "content": "A 12, 中英 mix."}]
    details = aligner.align_chat_messages(messages, return_details=True)

    assert len(details["slm_ids_padded"]) == len(details["llm_ids_padded"])
    assert any(details["message_mask"])
    for token_id, is_message in zip(details["llm_ids_padded"], details["message_mask"]):
        if is_message:
            assert token_id != llm_tokenizer.unk_token_id


def test_span_overlap_keeps_template_sections_structural():
    slm_tokenizer = FakeTokenizer("char")
    llm_tokenizer = FakeTokenizer("word")
    aligner = TokenAligner(
        slm_tokenizer,
        llm_tokenizer,
        strategy=AlignmentStrategy.SPAN_OVERLAP,
    )

    messages = [{"role": "user", "content": "hello"}]
    details = aligner.align_chat_messages(messages, return_details=True)

    template_sections = [
        section for section in details["sections"] if section["type"] == "template"
    ]
    message_sections = [
        section for section in details["sections"] if section["type"] == "message"
    ]

    assert template_sections
    assert message_sections
    for section in template_sections:
        start, end = section["slm_range"]
        assert not any(details["message_mask"][start:end])
    for section in message_sections:
        start, end = section["slm_range"]
        assert all(details["message_mask"][start:end])


def test_aligned_chat_dataset_truncates_masks_labels_and_collates():
    slm_tokenizer = FakeTokenizer("char")
    llm_tokenizer = FakeTokenizer("word")
    aligner = TokenAligner(
        slm_tokenizer,
        llm_tokenizer,
        strategy=AlignmentStrategy.SPAN_OVERLAP,
    )
    raw_dataset = [
        [
            {"role": "user", "content": "Question: " + ("abc " * 40)},
            {"role": "assistant", "content": "The correct answer is A."},
        ],
        [
            {"role": "user", "content": "Question: " + ("xyz " * 40)},
            {"role": "assistant", "content": "The correct answer is B."},
        ],
    ]
    dataset = AlignedChatDataset(raw_dataset, aligner, max_length=32)

    first = dataset[0]

    assert len(first["input_ids"][0]) == 32
    assert len(first["input_ids"][1]) == 32
    assert len(first["labels"]) == 32
    assert len(first["model_padding_mask"][0]) == 32
    assert len(first["model_padding_mask"][1]) == 32
    assert first["kv_cache_index"].shape == (32, 2)

    collator = RosettaDataCollator(
        slm_tokenizer=slm_tokenizer,
        llm_tokenizer=llm_tokenizer,
        max_length=32,
        aligner=aligner,
        do_alignment=True,
    )
    batch = collator([dataset[0], dataset[1]])

    assert batch["input_ids"][0].shape[0] == 2
    assert batch["input_ids"][1].shape[0] == 2
    assert batch["input_ids"][0].shape[1] <= 32
    assert batch["input_ids"][1].shape[1] <= 32
    assert batch["labels"].shape[1] <= 32


def test_soft_span_overlap_returns_topk_weights_and_matches_hard_top1():
    slm_tokenizer = FakeTokenizer("char")
    llm_tokenizer = FakeTokenizer("word")
    hard = TokenAligner(
        slm_tokenizer,
        llm_tokenizer,
        strategy=AlignmentStrategy.SPAN_OVERLAP,
    )
    soft = TokenAligner(
        slm_tokenizer,
        llm_tokenizer,
        strategy=AlignmentStrategy.SOFT_SPAN_OVERLAP,
    )

    messages = [{"role": "user", "content": "alpha beta, gamma"}]
    hard_details = hard.align_chat_messages(messages, return_details=True)
    soft_details = soft.align_chat_messages_soft(messages, return_details=True, top_k=3)

    assert len(soft_details["slm_ids"]) != len(soft_details["llm_ids"])
    assert len(soft_details["soft_alignment"]["source_indices"]) == len(
        soft_details["slm_ids"]
    )

    for i, is_message in enumerate(soft_details["message_mask"]):
        indices = soft_details["soft_alignment"]["source_indices"][i]
        weights = soft_details["soft_alignment"]["source_weights"][i]
        if not is_message:
            assert indices == [-1, -1, -1]
            assert weights == [0.0, 0.0, 0.0]
            continue

        assert abs(sum(weights) - 1.0) < 1e-6
        assert indices[0] >= 0
        assert soft_details["llm_ids"][indices[0]] == hard_details["llm_ids_padded"][i]


def test_soft_aligned_chat_dataset_collates_independent_source_length():
    slm_tokenizer = FakeTokenizer("char")
    llm_tokenizer = FakeTokenizer("word")
    aligner = TokenAligner(
        slm_tokenizer,
        llm_tokenizer,
        strategy=AlignmentStrategy.SOFT_SPAN_OVERLAP,
    )
    raw_dataset = [
        [
            {
                "role": "user",
                "content": "Question: " + ("abcdefghijklmnopqrstuvwxyz" * 8),
            },
            {"role": "assistant", "content": "The correct answer is A."},
        ],
        [
            {
                "role": "user",
                "content": "Question: " + ("zyxwvutsrqponmlkjihgfedcba" * 8),
            },
            {"role": "assistant", "content": "The correct answer is B."},
        ],
    ]
    dataset = AlignedChatDataset(
        raw_dataset,
        aligner,
        max_length=96,
        soft_alignment_top_k=4,
    )

    first = dataset[0]

    assert "soft_alignment" in first
    assert len(first["input_ids"][0]) == 96
    assert len(first["input_ids"][1]) < len(first["input_ids"][0])
    assert first["soft_alignment"]["source_indices"].shape == (96, 4)
    assert first["soft_alignment"]["source_weights"].shape == (96, 4)
    assert first["soft_alignment"]["source_confidence"].shape == (96,)
    assert first["soft_alignment"]["source_entropy"].shape == (96,)
    assert first["soft_alignment"]["source_entropy_override"].shape == (96,)
    assert not first["soft_alignment"]["source_entropy_override"].any()

    collator = RosettaDataCollator(
        slm_tokenizer=slm_tokenizer,
        llm_tokenizer=llm_tokenizer,
        max_length=96,
        aligner=aligner,
        do_alignment=True,
    )
    batch = collator([dataset[0], dataset[1]])

    assert len(batch["input_ids"]) == 2
    assert batch["input_ids"][0].shape[1] == 96
    assert batch["input_ids"][1].shape[1] < batch["input_ids"][0].shape[1]
    assert len(batch["soft_alignment"]) == len(batch["kv_cache_index"])
    for section in batch["soft_alignment"]:
        assert section["source_indices"].shape[-1] == 4
        assert section["source_weights"].shape[-1] == 4
        assert section["source_confidence"].dim() == 2
        assert section["source_entropy"].dim() == 2
        assert section["source_entropy_override"].dim() == 2
        assert section["source_entropy_override"].dtype == torch.bool


def test_soft_aligned_chat_dataset_collates_candidate_replay_cache(tmp_path):
    slm_tokenizer = FakeTokenizer("char")
    llm_tokenizer = FakeTokenizer("word")
    aligner = TokenAligner(
        slm_tokenizer,
        llm_tokenizer,
        strategy=AlignmentStrategy.LEARNED_SPAN_ALIGNMENT,
    )
    raw_dataset = [
        [
            {"role": "user", "content": "Question: alpha"},
            {"role": "assistant", "content": "The correct answer is A."},
        ],
        [
            {"role": "user", "content": "Question: beta"},
            {"role": "assistant", "content": "The correct answer is B."},
        ],
    ]
    cache_path = tmp_path / "candidate_cache.jsonl"
    cache_path.write_text(
        json.dumps(
            {
                "idx": 0,
                "cached": True,
                "target": [0.05, 0.9, 0.05],
                "utility": [0.0, 1.0, -1.0],
                "utility_valid": [True, True, False],
            }
        )
        + "\n"
    )
    dataset = AlignedChatDataset(
        raw_dataset,
        aligner,
        max_length=64,
        soft_alignment_top_k=3,
        candidate_replay_cache_path=str(cache_path),
    )

    assert "candidate_replay" in dataset[0]
    assert "candidate_replay" not in dataset[1]

    collator = RosettaDataCollator(
        slm_tokenizer=slm_tokenizer,
        llm_tokenizer=llm_tokenizer,
        max_length=64,
        aligner=aligner,
        do_alignment=True,
    )
    batch = collator([dataset[0], dataset[1]])

    replay = batch["candidate_replay"]
    assert replay["target"].shape == (2, 3)
    assert replay["utility"].shape == (2, 3)
    assert replay["utility_valid"].shape == (2, 3)
    assert replay["cache_hit"].tolist() == [True, False]
    assert torch.allclose(replay["target"][0], torch.tensor([0.05, 0.9, 0.05]))
    assert torch.all(replay["target"][1] == 0)
    assert replay["utility_valid"][0].tolist() == [True, True, False]
    assert replay["utility_valid"][1].tolist() == [False, False, False]


def test_cached_candidate_replay_target_filters_by_hit_and_margin():
    batch = {
        "candidate_replay": {
            "target": torch.tensor(
                [
                    [0.1, 0.8, 0.1],
                    [0.34, 0.33, 0.33],
                    [0.0, 0.0, 0.0],
                ]
            ),
            "utility": torch.tensor(
                [
                    [0.0, 1.0, -0.5],
                    [0.0, 0.01, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            "utility_valid": torch.tensor(
                [
                    [True, True, False],
                    [True, True, True],
                    [True, True, True],
                ]
            ),
            "cache_hit": torch.tensor([True, True, False]),
        }
    }

    target, utility, utility_valid, metrics = (
        _cached_candidate_replay_target_from_batch(
            batch,
            device="cpu",
            config={"min_target_margin": 0.2, "min_utility_margin": 0.2},
        )
    )

    assert target is not None
    assert utility is not None
    assert utility_valid is not None
    assert torch.allclose(target[0], torch.tensor([0.1, 0.8, 0.1]))
    assert torch.all(target[1] == 0)
    assert torch.all(target[2] == 0)
    assert utility_valid.tolist() == [
        [True, True, False],
        [False, False, False],
        [False, False, False],
    ]
    assert metrics["candidate_replay/cache_hit_rate"] == pytest.approx(2 / 3)
    assert metrics["candidate_replay/cache_selected_rate"] == pytest.approx(1 / 3)
    assert metrics["candidate_replay/best_rank"] == pytest.approx(1.0)


def test_c2c_projector_cached_replay_utility_valid_masks_candidates():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_aux_loss_mode="grad_ce_margin_rank",
        learned_alignment_margin_rank_loss_weight=0.1,
    )
    projector.set_learned_alignment_replay_utility(torch.tensor([[0.0, 1.0, 100.0]]))
    projector.set_learned_alignment_replay_utility_valid(
        torch.tensor([[True, True, False]])
    )

    utility_state = projector._expand_replay_utility_for_candidates(
        weights=torch.full((1, 2, 3), 1 / 3),
        valid_mask=torch.ones(1, 2, 3, dtype=torch.bool),
    )

    assert utility_state is not None
    utility, utility_valid = utility_state
    assert utility.shape == (1, 2, 3)
    assert utility_valid.tolist() == [[[True, True, False], [True, True, False]]]


def test_c2c_projector_cached_replay_target_expands_and_normalizes():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_aux_loss_mode="grad_ce_margin_rank",
        learned_alignment_aux_loss_weight=0.1,
    )
    projector.set_learned_alignment_replay_target(torch.tensor([[0.0, 2.0, 1.0]]))

    target_state = projector._cached_replay_target_for_candidates(
        weights=torch.full((1, 2, 3), 1 / 3),
        valid_mask=torch.tensor([[[True, True, False], [True, True, True]]]),
    )

    assert target_state is not None
    target, selected = target_state
    assert selected.tolist() == [[True, True]]
    assert torch.allclose(target[0, 0], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(target[0, 1], torch.tensor([0.0, 2 / 3, 1 / 3]))


def test_load_projector_checkpoint_dir_restores_projector_state(tmp_path):
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
    )
    checkpoint_dir = tmp_path / "projectors"
    checkpoint_dir.mkdir()

    with torch.no_grad():
        projector.key_gate_logit.fill_(1.25)
    torch.save(projector.state_dict(), checkpoint_dir / "projector_0.pt")

    restored = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
    )
    assert restored.key_gate_logit.item() != pytest.approx(1.25)

    _load_projector_checkpoint_dir([restored], str(checkpoint_dir))

    assert restored.key_gate_logit.item() == pytest.approx(1.25)


def test_weighted_source_kv_from_indices_gathers_and_normalizes():
    source_key = torch.tensor([[[[0.0], [1.0], [2.0], [3.0]]]])
    source_value = torch.tensor([[[[10.0], [11.0], [12.0], [13.0]]]])
    source_indices = torch.tensor([[[0, 2], [1, -1], [9, -1]]])
    source_weights = torch.tensor([[[0.25, 0.75], [1.0, 1.0], [0.5, 0.5]]])

    key, value = RosettaModel._weighted_source_kv_from_indices(
        source_key,
        source_value,
        source_indices,
        source_weights,
    )

    assert key.shape == (1, 1, 3, 1)
    assert value.shape == (1, 1, 3, 1)
    assert torch.allclose(key[0, 0, :, 0], torch.tensor([1.5, 1.0, 0.0]))
    assert torch.allclose(value[0, 0, :, 0], torch.tensor([11.5, 11.0, 0.0]))


def test_source_confidence_scales_projector_residual():
    base_key = torch.tensor([[[[10.0], [20.0], [30.0]]]])
    base_value = torch.tensor([[[[100.0], [200.0], [300.0]]]])
    projected_key = torch.tensor([[[[20.0], [40.0], [60.0]]]])
    projected_value = torch.tensor([[[[110.0], [240.0], [390.0]]]])

    key, value = RosettaModel._apply_source_confidence_to_projected_kv(
        projected_key=projected_key,
        projected_value=projected_value,
        base_key=base_key,
        base_value=base_value,
        soft_section={"source_confidence": torch.tensor([[0.0, 0.5, 1.0]])},
    )

    assert torch.allclose(key[0, 0, :, 0], torch.tensor([10.0, 30.0, 60.0]))
    assert torch.allclose(value[0, 0, :, 0], torch.tensor([100.0, 220.0, 390.0]))


def test_c2c_projector_span_weight_calibration_starts_noop():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        alignment_weight_calibration_mode="span_mlp",
    )

    source_weights = torch.tensor(
        [[[0.25, 0.75, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    source_indices = torch.tensor([[[0, 2, -1], [1, -1, -1], [-1, -1, -1]]])

    calibrated = projector.calibrate_source_weights(
        source_weights=source_weights,
        source_indices=source_indices,
    )

    expected = torch.tensor([[[0.25, 0.75, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    assert torch.allclose(calibrated, expected, atol=1e-6)
    assert calibrated.dtype == source_weights.dtype

    projector.to(dtype=torch.bfloat16)
    calibrated_bf16 = projector.calibrate_source_weights(
        source_weights=source_weights.to(dtype=torch.bfloat16),
        source_indices=source_indices,
    )
    assert projector.alignment_weight_calibration_head.weight.dtype == torch.float32
    assert calibrated_bf16.dtype == torch.bfloat16
    assert torch.allclose(calibrated_bf16.float(), expected, atol=1e-3)


def test_c2c_projector_span_weight_calibration_can_learn_rank_bias():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        alignment_weight_calibration_mode="span_mlp",
        alignment_weight_calibration_max_delta=4.0,
    )

    source_weights = torch.tensor([[[0.5, 0.5, 0.0]]])
    source_indices = torch.tensor([[[3, 4, -1]]])
    before = projector.calibrate_source_weights(source_weights, source_indices)

    with torch.no_grad():
        # Feature index 2 is rank_feature: 1.0 for top rank, 0.0 for last rank.
        projector.alignment_weight_calibration_head.weight[0, 2] = 2.0

    after = projector.calibrate_source_weights(source_weights, source_indices)

    assert torch.allclose(before, torch.tensor([[[0.5, 0.5, 0.0]]]), atol=1e-6)
    assert after[0, 0, 0] > before[0, 0, 0]
    assert after[0, 0, 1] < before[0, 0, 1]
    assert after[0, 0, 2] == 0.0
    assert torch.allclose(after.sum(dim=-1), torch.ones(1, 1), atol=1e-6)


def test_c2c_projector_span_weight_calibration_can_target_ambiguous_rows():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        alignment_weight_calibration_mode="span_mlp",
        alignment_weight_calibration_apply_mode="ambiguous",
        alignment_weight_calibration_entropy_threshold=0.5,
        alignment_weight_calibration_max_delta=4.0,
    )

    source_weights = torch.tensor([[[0.99, 0.01], [0.5, 0.5]]])
    source_indices = torch.tensor([[[3, 4], [5, 6]]])

    with torch.no_grad():
        # Feature index 2 is rank_feature: 1.0 for top rank, 0.0 for last rank.
        projector.alignment_weight_calibration_head.weight[0, 2] = 2.0

    calibrated = projector.calibrate_source_weights(source_weights, source_indices)

    assert torch.allclose(calibrated[0, 0], source_weights[0, 0], atol=1e-6)
    assert calibrated[0, 1, 0] > source_weights[0, 1, 0]
    assert calibrated[0, 1, 1] < source_weights[0, 1, 1]
    assert torch.allclose(calibrated.sum(dim=-1), torch.ones(1, 2), atol=1e-6)
    assert projector.last_alignment_weight_calibration_selected_rate == 0.5


def test_c2c_projector_span_weight_calibration_regularization_backprops():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        alignment_weight_calibration_mode="span_mlp",
        alignment_weight_calibration_apply_mode="ambiguous",
        alignment_weight_calibration_delta_l2_weight=0.1,
        alignment_weight_calibration_entropy_l2_weight=0.1,
        alignment_weight_calibration_max_delta=4.0,
    )

    source_weights = torch.tensor([[[0.5, 0.5]]])
    source_indices = torch.tensor([[[3, 4]]])
    with torch.no_grad():
        projector.alignment_weight_calibration_head.weight[0, 2] = 2.0

    projector.calibrate_source_weights(source_weights, source_indices)
    aux_loss = projector.alignment_regularization_loss()

    assert aux_loss is not None
    assert aux_loss.requires_grad
    assert aux_loss.item() > 0

    aux_loss.backward()
    assert projector.alignment_weight_calibration_head.weight.grad is not None
    assert torch.any(projector.alignment_weight_calibration_head.weight.grad != 0)


def test_c2c_projector_token_confidence_gate_starts_from_source_confidence():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=2,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        alignment_confidence_gate_mode="token_mlp",
    )

    key_hidden = torch.zeros(1, 3, 4)
    value_hidden = torch.zeros(1, 3, 4)
    source_confidence = torch.tensor([[0.25, 0.5, 0.75]])
    source_weights = torch.tensor([[[1.0, 0.0], [0.5, 0.5], [1.0, 0.0]]])

    key_confidence, value_confidence = projector._compute_alignment_confidence(
        source_confidence=source_confidence,
        source_weights=source_weights,
        key_hidden=key_hidden,
        value_hidden=value_hidden,
        target_shape=(1, 2, 3, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    expected = source_confidence[:, None, :, None].expand(1, 2, 3, 1)
    assert projector.uses_internal_source_confidence()
    assert torch.allclose(key_confidence, expected, atol=1e-6)
    assert torch.allclose(value_confidence, expected, atol=1e-6)

    with torch.no_grad():
        projector.key_alignment_confidence_head.bias.fill_(1.0)

    key_confidence_after, value_confidence_after = (
        projector._compute_alignment_confidence(
            source_confidence=source_confidence,
            source_weights=source_weights,
            key_hidden=key_hidden,
            value_hidden=value_hidden,
            target_shape=(1, 2, 3, 2),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    )

    assert torch.all(key_confidence_after > key_confidence)
    assert torch.allclose(value_confidence_after, value_confidence, atol=1e-6)


def test_c2c_projector_quality_confidence_gate_starts_from_source_confidence():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=2,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        alignment_confidence_gate_mode="token_mlp",
        alignment_confidence_feature_mode="quality",
    )

    key_hidden = torch.zeros(1, 3, 4)
    value_hidden = torch.zeros(1, 3, 4)
    source_confidence = torch.tensor([[0.25, 0.5, 0.75]])
    source_weights = torch.tensor([[[1.0, 0.0], [0.5, 0.5], [1.0, 0.0]]])

    key_confidence, value_confidence = projector._compute_alignment_confidence(
        source_confidence=source_confidence,
        source_weights=source_weights,
        key_hidden=key_hidden,
        value_hidden=value_hidden,
        target_shape=(1, 2, 3, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    expected = source_confidence[:, None, :, None].expand(1, 2, 3, 1)
    assert torch.allclose(key_confidence, expected, atol=1e-6)
    assert torch.allclose(value_confidence, expected, atol=1e-6)
    assert projector.last_alignment_quality_entropy_mean > 0.0
    assert projector.last_alignment_quality_top1_mean < 1.0


def test_c2c_projector_quality_confidence_gate_can_use_entropy_feature():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        alignment_confidence_gate_mode="token_mlp",
        alignment_confidence_feature_mode="quality",
    )

    key_hidden = torch.zeros(1, 2, 4)
    value_hidden = torch.zeros(1, 2, 4)
    source_confidence = torch.tensor([[0.5, 0.5]])
    source_weights = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])

    with torch.no_grad():
        entropy_feature_index = 4 + 1
        projector.key_alignment_confidence_head.weight[0, entropy_feature_index] = 2.0

    key_confidence, value_confidence = projector._compute_alignment_confidence(
        source_confidence=source_confidence,
        source_weights=source_weights,
        key_hidden=key_hidden,
        value_hidden=value_hidden,
        target_shape=(1, 1, 2, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert key_confidence[0, 0, 1, 0] > key_confidence[0, 0, 0, 0]
    assert torch.allclose(
        value_confidence,
        source_confidence[:, None, :, None],
        atol=1e-6,
    )


def test_c2c_projector_explicit_entropy_override_blocks_source_weight_leakage():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        alignment_confidence_gate_mode="token_mlp",
    )
    source_confidence = torch.tensor([[0.5, 0.5]])
    source_weights = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])
    common_kwargs = {
        "source_confidence": source_confidence,
        "source_weights": source_weights,
        "key_hidden": torch.zeros(1, 2, 4),
        "value_hidden": torch.zeros(1, 2, 4),
        "target_shape": (1, 1, 2, 2),
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }
    with torch.no_grad():
        projector.key_alignment_entropy_scale.fill_(2.0)
        projector.value_alignment_entropy_scale.fill_(2.0)

    native_key, native_value = projector._compute_alignment_confidence(
        **common_kwargs
    )
    native_with_observation_key, native_with_observation_value = (
        projector._compute_alignment_confidence(
            **common_kwargs,
            source_entropy=torch.tensor([[1.0, 0.0]]),
            source_entropy_override=torch.tensor([[False, False]]),
        )
    )
    controlled_key, controlled_value = projector._compute_alignment_confidence(
        **common_kwargs,
        source_entropy=torch.tensor([[1.0, 0.0]]),
        source_entropy_override=torch.tensor([[True, True]]),
    )

    assert torch.equal(native_key, native_with_observation_key)
    assert torch.equal(native_value, native_with_observation_value)
    assert native_key[0, 0, 0, 0] < native_key[0, 0, 1, 0]
    assert controlled_key[0, 0, 0, 0] > controlled_key[0, 0, 1, 0]
    assert controlled_value[0, 0, 0, 0] > controlled_value[0, 0, 1, 0]


def test_c2c_projector_token_confidence_regularization_is_trainable():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=2,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        alignment_confidence_gate_mode="token_mlp",
        alignment_confidence_delta_l2_weight=0.1,
    )

    key_hidden = torch.ones(1, 3, 4)
    value_hidden = torch.ones(1, 3, 4)
    source_confidence = torch.tensor([[0.25, 0.5, 0.75]])
    source_weights = torch.tensor([[[1.0, 0.0], [0.5, 0.5], [1.0, 0.0]]])

    with torch.no_grad():
        projector.key_alignment_confidence_head.bias.fill_(0.5)

    projector._compute_alignment_confidence(
        source_confidence=source_confidence,
        source_weights=source_weights,
        key_hidden=key_hidden,
        value_hidden=value_hidden,
        target_shape=(1, 2, 3, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    aux_loss = projector.alignment_regularization_loss()

    assert aux_loss is not None
    assert aux_loss.requires_grad
    assert aux_loss.item() > 0

    aux_loss.backward()

    assert projector.key_alignment_confidence_head.bias.grad is not None
    assert torch.any(projector.key_alignment_confidence_head.bias.grad != 0)


def test_c2c_projector_selective_regularization_skips_confident_rows():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=2,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        alignment_confidence_gate_mode="token_mlp",
        alignment_confidence_delta_l2_weight=0.1,
        alignment_confidence_delta_l2_mode="uncertain",
        alignment_confidence_delta_l2_confidence_threshold=0.99,
    )

    key_hidden = torch.ones(1, 2, 4)
    value_hidden = torch.ones(1, 2, 4)
    source_weights = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    with torch.no_grad():
        projector.key_alignment_confidence_head.bias.fill_(0.5)
        projector.value_alignment_confidence_head.bias.fill_(0.25)

    projector._compute_alignment_confidence(
        source_confidence=torch.tensor([[1.0, 1.0]]),
        source_weights=source_weights,
        key_hidden=key_hidden,
        value_hidden=value_hidden,
        target_shape=(1, 2, 2, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    confident_aux_loss = projector.alignment_regularization_loss()

    assert confident_aux_loss is not None
    assert confident_aux_loss.item() == 0.0
    assert projector.last_alignment_regularization_selected_rate == 0.0

    projector._compute_alignment_confidence(
        source_confidence=torch.tensor([[1.0, 0.5]]),
        source_weights=source_weights,
        key_hidden=key_hidden,
        value_hidden=value_hidden,
        target_shape=(1, 2, 2, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    uncertain_aux_loss = projector.alignment_regularization_loss()

    assert uncertain_aux_loss is not None
    assert uncertain_aux_loss.requires_grad
    assert uncertain_aux_loss.item() > 0
    assert projector.last_alignment_regularization_selected_rate == 0.5


def test_c2c_projector_layer_scale_modulates_token_confidence_delta():
    common_kwargs = {
        "source_dim": 2,
        "target_dim": 2,
        "source_num_heads": 1,
        "target_num_heads": 1,
        "hidden_dim": 4,
        "intermediate_dim": 4,
        "num_layers": 3,
        "dropout": 0.0,
        "alignment_confidence_gate_mode": "token_mlp",
        "alignment_confidence_max_delta": 0.0,
        "alignment_confidence_layer_scale_mode": "early_key_late_value",
        "alignment_confidence_num_layers": 4,
        "alignment_confidence_key_layer_scale_start": 2.0,
        "alignment_confidence_key_layer_scale_end": 0.5,
        "alignment_confidence_value_layer_scale_start": 0.5,
        "alignment_confidence_value_layer_scale_end": 2.0,
    }
    early_projector = C2CProjector(
        **common_kwargs,
        alignment_confidence_layer_idx=0,
    )
    late_projector = C2CProjector(
        **common_kwargs,
        alignment_confidence_layer_idx=3,
    )

    for projector in (early_projector, late_projector):
        with torch.no_grad():
            projector.key_alignment_confidence_head.bias.fill_(1.0)
            projector.value_alignment_confidence_head.bias.fill_(1.0)

    kwargs = {
        "source_confidence": torch.tensor([[0.5]]),
        "source_weights": torch.tensor([[[1.0, 0.0]]]),
        "key_hidden": torch.zeros(1, 1, 4),
        "value_hidden": torch.zeros(1, 1, 4),
        "target_shape": (1, 1, 1, 2),
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }
    early_key_confidence, early_value_confidence = (
        early_projector._compute_alignment_confidence(**kwargs)
    )
    late_key_confidence, late_value_confidence = (
        late_projector._compute_alignment_confidence(**kwargs)
    )

    assert early_projector.last_alignment_key_layer_scale == 2.0
    assert early_projector.last_alignment_value_layer_scale == 0.5
    assert late_projector.last_alignment_key_layer_scale == 0.5
    assert late_projector.last_alignment_value_layer_scale == 2.0
    assert torch.all(early_key_confidence > late_key_confidence)
    assert torch.all(late_value_confidence > early_value_confidence)


def test_c2c_projector_learned_layer_scale_is_trainable():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        alignment_confidence_gate_mode="token_mlp",
        alignment_confidence_max_delta=0.0,
        alignment_confidence_layer_scale_mode="learned",
        alignment_confidence_key_layer_scale_init=1.0,
        alignment_confidence_value_layer_scale_init=1.0,
    )

    with torch.no_grad():
        projector.key_alignment_confidence_head.bias.fill_(1.0)
        projector.value_alignment_confidence_head.bias.fill_(1.0)

    key_confidence, value_confidence = projector._compute_alignment_confidence(
        source_confidence=torch.tensor([[0.5]]),
        source_weights=torch.tensor([[[1.0, 0.0]]]),
        key_hidden=torch.zeros(1, 1, 4),
        value_hidden=torch.zeros(1, 1, 4),
        target_shape=(1, 1, 1, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    loss = key_confidence.mean() + value_confidence.mean()
    loss.backward()

    assert projector.last_alignment_key_layer_scale == 1.0
    assert projector.last_alignment_value_layer_scale == 1.0
    assert projector.alignment_confidence_key_token_delta_scale_param.grad is not None
    assert projector.alignment_confidence_value_token_delta_scale_param.grad is not None
    assert torch.any(
        projector.alignment_confidence_key_token_delta_scale_param.grad != 0
    )
    assert torch.any(
        projector.alignment_confidence_value_token_delta_scale_param.grad != 0
    )


def test_c2c_projector_learned_layer_scale_uses_fp32_parameters():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        dtype=torch.bfloat16,
        alignment_confidence_gate_mode="token_mlp",
        alignment_confidence_layer_scale_mode="learned",
    )
    projector.to(dtype=torch.bfloat16)

    assert (
        projector.alignment_confidence_key_token_delta_scale_param.dtype
        == torch.float32
    )
    assert (
        projector.alignment_confidence_value_token_delta_scale_param.dtype
        == torch.float32
    )


def test_c2c_projector_learned_residual_scale_modulates_projection():
    torch.manual_seed(0)
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        alignment_residual_scale_mode="learned",
        alignment_residual_scale_max_delta=1.0,
    )
    projector.eval()
    with torch.no_grad():
        projector.key_gate_logit.fill_(1.0)
        projector.value_gate_logit.fill_(1.0)

    source_key = torch.randn(1, 1, 2, 2)
    source_value = torch.randn(1, 1, 2, 2)
    target_key = torch.randn(1, 1, 2, 2)
    target_value = torch.randn(1, 1, 2, 2)

    key_before, value_before = projector(
        (source_key, source_value),
        (target_key, target_value),
    )
    assert projector.last_alignment_key_residual_scale == 1.0
    assert projector.last_alignment_value_residual_scale == 1.0

    with torch.no_grad():
        projector.alignment_residual_key_scale_delta.fill_(0.5)
        projector.alignment_residual_value_scale_delta.fill_(-0.5)

    key_after, value_after = projector(
        (source_key, source_value),
        (target_key, target_value),
    )

    assert projector.last_alignment_key_residual_scale > 1.0
    assert projector.last_alignment_value_residual_scale < 1.0
    assert torch.linalg.vector_norm(key_after - target_key) > torch.linalg.vector_norm(
        key_before - target_key
    )
    assert torch.linalg.vector_norm(
        value_after - target_value
    ) < torch.linalg.vector_norm(value_before - target_value)


def test_c2c_projector_learned_residual_scale_regularization_backprops():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        alignment_residual_scale_mode="learned",
        alignment_residual_scale_max_delta=1.0,
        alignment_residual_scale_l2_weight=0.1,
    )
    with torch.no_grad():
        projector.alignment_residual_key_scale_delta.fill_(0.5)
        projector.alignment_residual_value_scale_delta.fill_(-0.25)

    projector._current_alignment_residual_scales(
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    aux_loss = projector.alignment_regularization_loss()

    assert aux_loss is not None
    assert aux_loss.requires_grad
    assert aux_loss.item() > 0

    aux_loss.backward()

    assert projector.alignment_residual_key_scale_delta.grad is not None
    assert projector.alignment_residual_value_scale_delta.grad is not None
    assert torch.any(projector.alignment_residual_key_scale_delta.grad != 0)
    assert torch.any(projector.alignment_residual_value_scale_delta.grad != 0)


def test_c2c_projector_learned_residual_scale_uses_fp32_parameters():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        dtype=torch.bfloat16,
        alignment_residual_scale_mode="learned",
    )
    projector.to(dtype=torch.bfloat16)

    assert projector.alignment_residual_key_scale_delta.dtype == torch.float32
    assert projector.alignment_residual_value_scale_delta.dtype == torch.float32


def test_c2c_projector_learned_affine_confidence_can_penalize_entropy():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        alignment_confidence_gate_mode="learned_affine",
    )

    with torch.no_grad():
        projector.key_alignment_entropy_scale.fill_(-1.0)
        projector.value_alignment_entropy_scale.fill_(-1.0)

    key_confidence, value_confidence = projector._compute_alignment_confidence(
        source_confidence=torch.tensor([[0.5, 0.5]]),
        source_weights=torch.tensor([[[1.0, 0.0], [0.5, 0.5]]]),
        key_hidden=torch.zeros(1, 2, 4),
        value_hidden=torch.zeros(1, 2, 4),
        target_shape=(1, 1, 2, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert key_confidence[0, 0, 1, 0] < key_confidence[0, 0, 0, 0]
    assert value_confidence[0, 0, 1, 0] < value_confidence[0, 0, 0, 0]


def test_soft_span_overlap_v2_uniform_weights_selected_candidates_equally():
    aligner = TokenAligner(
        FakeTokenizer("char"),
        FakeTokenizer("char"),
        strategy=AlignmentStrategy.SOFT_SPAN_OVERLAP_V2,
        soft_alignment_score_mode="uniform",
    )

    rows = aligner._soft_align_message_by_span_overlap(
        slm_ids=[10],
        slm_offsets=[(0, 4)],
        slm_range=(0, 1),
        slm_span=(0, 4),
        llm_offsets=[(0, 1), (1, 4)],
        llm_range=(0, 2),
        llm_span=(0, 4),
        top_k=2,
    )

    assert rows["source_indices"] == [[1, 0]]
    assert rows["source_weights"] == [[0.5, 0.5]]
    assert rows["source_confidence"] == [1.0]
    assert rows["positive_overlap_counts"] == [2]


def test_soft_span_overlap_v2_entropy_confidence_penalizes_ambiguous_rows():
    aligner = TokenAligner(
        FakeTokenizer("char"),
        FakeTokenizer("char"),
        strategy=AlignmentStrategy.SOFT_SPAN_OVERLAP_V2,
        soft_alignment_score_mode="uniform",
        soft_alignment_confidence_mode="entropy",
        soft_alignment_confidence_alpha=0.5,
        soft_alignment_confidence_floor=0.25,
        soft_alignment_fallback_confidence=0.1,
    )

    rows = aligner._soft_align_message_by_span_overlap(
        slm_ids=[10, 11],
        slm_offsets=[(0, 4), (5, 6)],
        slm_range=(0, 2),
        slm_span=(0, 6),
        llm_offsets=[(0, 2), (2, 4)],
        llm_range=(0, 2),
        llm_span=(0, 6),
        top_k=2,
    )

    assert rows["source_weights"][0] == [0.5, 0.5]
    assert abs(rows["source_confidence"][0] - 0.5) < 1e-6
    assert rows["fallback_mask"][1] == [True][0]
    assert abs(rows["source_confidence"][1] - 0.1) < 1e-6


def test_constant_confidence_control_requires_explicit_value():
    with pytest.raises(ValueError, match="constant_value is required"):
        TokenAligner(
            FakeTokenizer("char"),
            FakeTokenizer("word"),
            strategy=AlignmentStrategy.SOFT_SPAN_OVERLAP_V2,
            soft_alignment_confidence_control_mode="constant",
        )


def test_constant_confidence_control_zeroes_projector_entropy_signal():
    aligner = TokenAligner(
        FakeTokenizer("char"),
        FakeTokenizer("word"),
        strategy=AlignmentStrategy.SOFT_SPAN_OVERLAP_V2,
        soft_alignment_score_mode="uniform",
        soft_alignment_confidence_mode="entropy",
        soft_alignment_confidence_control_mode="constant",
        soft_alignment_confidence_constant_value=0.93,
    )

    details = aligner.align_chat_messages_soft(
        [{"role": "user", "content": "alpha beta"}],
        add_generation_prompt=False,
        top_k=4,
    )
    soft = details["soft_alignment"]
    message_indices = [
        idx for idx, is_message in enumerate(details["message_mask"]) if is_message
    ]
    template_indices = [
        idx for idx, is_message in enumerate(details["message_mask"]) if not is_message
    ]

    assert message_indices
    assert all(soft["source_confidence"][idx] == pytest.approx(0.93) for idx in message_indices)
    assert all(soft["source_entropy"][idx] == 0.0 for idx in message_indices)
    assert all(soft["source_entropy_override"][idx] for idx in message_indices)
    assert all(soft["source_confidence"][idx] == 1.0 for idx in template_indices)
    assert not any(soft["source_entropy_override"][idx] for idx in template_indices)


def test_shuffle_confidence_control_is_deterministic_and_pair_preserving():
    aligner = TokenAligner(
        FakeTokenizer("char"),
        FakeTokenizer("word"),
        strategy=AlignmentStrategy.SOFT_SPAN_OVERLAP_V2,
        soft_alignment_confidence_control_mode="shuffle",
        soft_alignment_confidence_shuffle_seed=44,
    )
    confidence = [1.0, 0.2, 0.4, 0.6, 1.0]
    entropy = [0.0, 0.8, 0.6, 0.4, 0.0]
    message_mask = [False, True, True, True, False]
    token_ids = [10, 11, 12, 13, 14]

    first = aligner._apply_confidence_control(
        confidence.copy(), entropy.copy(), message_mask, token_ids
    )
    second = aligner._apply_confidence_control(
        confidence.copy(), entropy.copy(), message_mask, token_ids
    )

    assert first == second
    shuffled_confidence, shuffled_entropy, override = first
    assert list(zip(shuffled_confidence[1:4], shuffled_entropy[1:4])) != list(
        zip(confidence[1:4], entropy[1:4])
    )
    assert sorted(zip(shuffled_confidence[1:4], shuffled_entropy[1:4])) == sorted(
        zip(confidence[1:4], entropy[1:4])
    )
    assert shuffled_confidence[0] == confidence[0]
    assert shuffled_entropy[4] == entropy[4]
    assert override == message_mask


def test_shuffle_control_uses_only_truncated_active_prompt_and_ignores_answer():
    def build(mode: str, assistant: str):
        aligner = TokenAligner(
            FakeTokenizer("char"),
            FakeTokenizer("word"),
            strategy=AlignmentStrategy.SOFT_SPAN_OVERLAP_V2,
            soft_alignment_score_mode="uniform",
            soft_alignment_confidence_mode="entropy",
            soft_alignment_confidence_control_mode=mode,
            soft_alignment_confidence_shuffle_seed=44,
        )
        dataset = AlignedChatDataset(
            [[
                {"role": "user", "content": "alpha beta gamma delta"},
                {"role": "assistant", "content": assistant},
            ]],
            aligner,
            max_length=128,
            soft_alignment_top_k=4,
        )
        return dataset[0]

    native = build("native", "answer one")
    shuffled_a = build("shuffle", "answer one")
    shuffled_b = build("shuffle", "a completely different gold answer")

    def masks(item):
        soft = item["soft_alignment"]
        message = soft["source_weights"].sum(dim=-1) > 0
        active = item["kv_cache_index"][:, 0] == 1
        answer = message & ~active
        return active, answer

    native_active, native_answer = masks(native)
    active_a, answer_a = masks(shuffled_a)
    active_b, _answer_b = masks(shuffled_b)
    assert torch.equal(native_active, active_a)
    assert torch.equal(
        active_a.nonzero(as_tuple=False), active_b.nonzero(as_tuple=False)
    )
    assert native_answer.any() and answer_a.any()

    def pairs(item, mask):
        soft = item["soft_alignment"]
        return list(
            zip(
                soft["source_confidence"][mask].tolist(),
                soft["source_entropy"][mask].tolist(),
            )
        )

    def assert_pairs_close(left, right):
        assert len(left) == len(right)
        for left_pair, right_pair in zip(left, right):
            assert left_pair == pytest.approx(right_pair)

    assert_pairs_close(
        sorted(pairs(shuffled_a, active_a)),
        sorted(pairs(native, native_active)),
    )
    assert_pairs_close(pairs(shuffled_a, active_a), pairs(shuffled_b, active_b))
    assert_pairs_close(pairs(shuffled_a, answer_a), pairs(native, native_answer))
    override = shuffled_a["soft_alignment"]["source_entropy_override"]
    assert torch.equal(override, active_a)
    assert not override[answer_a].any()


def test_constant_control_keeps_all_message_token_semantics_in_dataset():
    aligner = TokenAligner(
        FakeTokenizer("char"),
        FakeTokenizer("word"),
        strategy=AlignmentStrategy.SOFT_SPAN_OVERLAP_V2,
        soft_alignment_confidence_control_mode="constant",
        soft_alignment_confidence_constant_value=0.93,
    )
    item = AlignedChatDataset(
        [[
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "gold answer"},
        ]],
        aligner,
        max_length=128,
        soft_alignment_top_k=4,
    )[0]
    soft = item["soft_alignment"]
    message = soft["source_weights"].sum(dim=-1) > 0

    assert message.any()
    assert torch.allclose(
        soft["source_confidence"][message],
        torch.full_like(soft["source_confidence"][message], 0.93),
    )
    assert not soft["source_entropy"][message].any()
    assert soft["source_entropy_override"][message].all()


def test_soft_span_overlap_v2_overlap_power2_emphasizes_large_overlap():
    aligner = TokenAligner(
        FakeTokenizer("char"),
        FakeTokenizer("char"),
        strategy=AlignmentStrategy.SOFT_SPAN_OVERLAP_V2,
        soft_alignment_score_mode="overlap_power2",
    )

    rows = aligner._soft_align_message_by_span_overlap(
        slm_ids=[10],
        slm_offsets=[(0, 4)],
        slm_range=(0, 1),
        slm_span=(0, 4),
        llm_offsets=[(0, 1), (1, 4)],
        llm_range=(0, 2),
        llm_span=(0, 4),
        top_k=2,
    )

    assert rows["source_indices"] == [[1, 0]]
    assert torch.allclose(
        torch.tensor(rows["source_weights"][0]),
        torch.tensor([0.9, 0.1]),
    )


def test_soft_span_overlap_v2_boundary_power2_adds_boundary_prior():
    aligner = TokenAligner(
        FakeTokenizer("char"),
        FakeTokenizer("char"),
        strategy=AlignmentStrategy.SOFT_SPAN_OVERLAP_V2,
        soft_alignment_score_mode="boundary_power2",
        soft_alignment_boundary_bonus=1.0,
        soft_alignment_boundary_tolerance=0,
    )

    rows = aligner._soft_align_message_by_span_overlap(
        slm_ids=[10],
        slm_offsets=[(0, 4)],
        slm_range=(0, 1),
        slm_span=(0, 4),
        llm_offsets=[(0, 2), (1, 3)],
        llm_range=(0, 2),
        llm_span=(0, 4),
        top_k=2,
    )

    assert rows["source_indices"] == [[0, 1]]
    assert rows["top1_boundary_hit_mask"] == [True]
    assert torch.allclose(
        torch.tensor(rows["source_weights"][0]),
        torch.tensor([2.0 / 3.0, 1.0 / 3.0]),
    )


def test_soft_span_overlap_v2_adaptive_overlap_reweights_uniform_rows():
    aligner = TokenAligner(
        FakeTokenizer("char"),
        FakeTokenizer("char"),
        strategy=AlignmentStrategy.SOFT_SPAN_OVERLAP_V2,
        soft_alignment_score_mode="uniform",
        soft_alignment_confidence_mode="entropy",
        soft_alignment_confidence_alpha=0.5,
        soft_alignment_reweight_mode="adaptive_overlap",
        soft_alignment_reweight_strength=1.0,
        soft_alignment_reweight_power=2.0,
    )

    rows = aligner._soft_align_message_by_span_overlap(
        slm_ids=[10],
        slm_offsets=[(0, 4)],
        slm_range=(0, 1),
        slm_span=(0, 4),
        llm_offsets=[(0, 1), (1, 4)],
        llm_range=(0, 2),
        llm_span=(0, 4),
        top_k=2,
    )

    weights = torch.tensor(rows["source_weights"][0])
    assert rows["source_indices"] == [[1, 0]]
    assert weights[0] > 0.5
    assert weights[1] < 0.5
    assert abs(float(weights.sum()) - 1.0) < 1e-6
    assert rows["source_confidence"][0] > 0.5


def test_soft_span_overlap_v2_adaptive_overlap_keeps_equal_overlap_uniform():
    aligner = TokenAligner(
        FakeTokenizer("char"),
        FakeTokenizer("char"),
        strategy=AlignmentStrategy.SOFT_SPAN_OVERLAP_V2,
        soft_alignment_score_mode="uniform",
        soft_alignment_reweight_mode="adaptive_overlap",
        soft_alignment_reweight_strength=1.0,
        soft_alignment_reweight_power=2.0,
    )

    rows = aligner._soft_align_message_by_span_overlap(
        slm_ids=[10],
        slm_offsets=[(0, 4)],
        slm_range=(0, 1),
        slm_span=(0, 4),
        llm_offsets=[(0, 2), (2, 4)],
        llm_range=(0, 2),
        llm_span=(0, 4),
        top_k=2,
    )

    assert rows["source_indices"] == [[0, 1]]
    assert torch.allclose(
        torch.tensor(rows["source_weights"][0]),
        torch.tensor([0.5, 0.5]),
    )


def test_learned_span_alignment_keeps_anchor_first_and_expands_neighbors():
    aligner = TokenAligner(
        FakeTokenizer("char"),
        FakeTokenizer("char"),
        strategy=AlignmentStrategy.LEARNED_SPAN_ALIGNMENT,
        soft_alignment_score_mode="uniform",
        soft_alignment_candidate_window=1,
    )

    rows = aligner._soft_align_message_by_span_overlap(
        slm_ids=[10],
        slm_offsets=[(0, 4)],
        slm_range=(0, 1),
        slm_span=(0, 4),
        llm_offsets=[(0, 1), (1, 4), (4, 5)],
        llm_range=(0, 3),
        llm_span=(0, 4),
        top_k=4,
    )

    assert rows["source_indices"] == [[1, 0, 2, -1]]
    assert rows["source_weights"] == [[1.0, 0.0, 0.0, 0.0]]
    assert rows["source_confidence"] == [1.0]
    assert rows["positive_overlap_counts"] == [2]


def test_learned_span_alignment_can_emit_soft_span_prior():
    aligner = TokenAligner(
        FakeTokenizer("char"),
        FakeTokenizer("char"),
        strategy=AlignmentStrategy.LEARNED_SPAN_ALIGNMENT,
        soft_alignment_score_mode="uniform",
        soft_alignment_candidate_window=1,
        learned_alignment_prior_mode="soft_span",
    )

    rows = aligner._soft_align_message_by_span_overlap(
        slm_ids=[10],
        slm_offsets=[(0, 4)],
        slm_range=(0, 1),
        slm_span=(0, 4),
        llm_offsets=[(0, 1), (1, 4), (4, 5)],
        llm_range=(0, 3),
        llm_span=(0, 4),
        top_k=4,
    )

    assert rows["source_indices"] == [[1, 0, 2, -1]]
    assert torch.allclose(
        torch.tensor(rows["source_weights"][0]),
        torch.tensor([0.5, 0.5, 0.0, 0.0]),
    )
    assert rows["source_confidence"] == [1.0]


def test_learned_span_alignment_entropy_confidence_can_match_v2_policy():
    aligner = TokenAligner(
        FakeTokenizer("char"),
        FakeTokenizer("char"),
        strategy=AlignmentStrategy.LEARNED_SPAN_ALIGNMENT,
        soft_alignment_score_mode="uniform",
        soft_alignment_confidence_mode="entropy",
        soft_alignment_confidence_alpha=0.5,
        soft_alignment_confidence_floor=0.5,
        soft_alignment_fallback_confidence=0.25,
        learned_alignment_prior_mode="soft_span",
    )

    rows = aligner._soft_align_message_by_span_overlap(
        slm_ids=[10, 11],
        slm_offsets=[(0, 4), (5, 6)],
        slm_range=(0, 2),
        slm_span=(0, 6),
        llm_offsets=[(0, 2), (2, 4)],
        llm_range=(0, 2),
        llm_span=(0, 6),
        top_k=2,
    )

    assert rows["source_weights"][0] == [0.5, 0.5]
    assert rows["source_confidence"][0] == pytest.approx(0.5)
    assert rows["fallback_mask"][1]
    assert rows["source_confidence"][1] == pytest.approx(0.25)


def test_learned_alignment_router_anchor_init_is_finite_and_masked():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=8,
        intermediate_dim=8,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_hidden_dim=4,
        learned_alignment_init="anchor",
        learned_alignment_anchor_logit=2.0,
    )

    source_key_candidates = torch.tensor(
        [[[[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]], [[3.0, 0.0], [9.0, 9.0], [8.0, 8.0]]]]]
    )
    source_value_candidates = source_key_candidates + 10.0
    target_key = torch.zeros(1, 1, 2, 2)
    target_value = torch.zeros(1, 1, 2, 2)
    valid_mask = torch.tensor([[[True, True, True], [True, False, False]]])

    key, value = projector.align_source_kv(
        source_kv_candidates=(source_key_candidates, source_value_candidates),
        target_kv=(target_key, target_value),
        valid_mask=valid_mask,
    )

    assert key.shape == (1, 1, 2, 2)
    assert value.shape == (1, 1, 2, 2)
    assert torch.isfinite(key).all()
    assert torch.isfinite(value).all()
    assert torch.allclose(key[0, 0, 1], source_key_candidates[0, 0, 1, 0])
    assert projector.last_learned_alignment_key_anchor_mean > 0.5
    assert projector.last_learned_alignment_value_anchor_mean > 0.5


def test_learned_alignment_aux_loss_mixes_span_anchor_with_valid_candidates():
    torch.manual_seed(0)
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=8,
        intermediate_dim=8,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_hidden_dim=4,
        learned_alignment_init="anchor",
        learned_alignment_anchor_logit=2.0,
        learned_alignment_aux_loss_mode="span_ce",
        learned_alignment_aux_loss_weight=0.1,
        learned_alignment_aux_apply_mode="ambiguous",
        learned_alignment_aux_uniform_mix=0.5,
    )

    source_key_candidates = torch.randn(1, 1, 2, 3, 2)
    source_value_candidates = torch.randn(1, 1, 2, 3, 2)
    target_key = torch.zeros(1, 1, 2, 2)
    target_value = torch.zeros(1, 1, 2, 2)
    valid_mask = torch.tensor([[[True, True, True], [True, False, False]]])
    source_weights = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])

    projector.align_source_kv(
        source_kv_candidates=(source_key_candidates, source_value_candidates),
        target_kv=(target_key, target_value),
        valid_mask=valid_mask,
        source_weights=source_weights,
    )
    aux_loss = projector.alignment_regularization_loss()

    assert aux_loss is not None
    assert aux_loss.requires_grad
    assert aux_loss.item() > 0
    assert projector.last_learned_alignment_aux_selected_rate == 0.5
    assert 0.0 < projector.last_learned_alignment_aux_target_entropy < 1.0
    assert 0.5 < projector.last_learned_alignment_aux_target_anchor < 1.0

    aux_loss.backward()

    assert projector.learned_key_alignment_score.weight.grad is not None
    assert projector.learned_value_alignment_score.weight.grad is not None
    assert torch.any(projector.learned_key_alignment_score.weight.grad != 0)
    assert torch.any(projector.learned_value_alignment_score.weight.grad != 0)


def test_learned_alignment_span_log_prior_starts_from_source_weights():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=8,
        intermediate_dim=8,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_hidden_dim=4,
        learned_alignment_init="uniform",
        learned_alignment_prior_mode="span_log_prior",
        learned_alignment_prior_strength=1.0,
    )

    source_key_candidates = torch.tensor([[[[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]]]])
    source_value_candidates = source_key_candidates + 10.0
    target_key = torch.zeros(1, 1, 1, 2)
    target_value = torch.zeros(1, 1, 1, 2)
    valid_mask = torch.tensor([[[True, True, True]]])
    source_weights = torch.tensor([[[0.2, 0.3, 0.5]]])

    key, value = projector.align_source_kv(
        source_kv_candidates=(source_key_candidates, source_value_candidates),
        target_kv=(target_key, target_value),
        valid_mask=valid_mask,
        source_weights=source_weights,
    )

    expected_key = (source_key_candidates * source_weights[:, None, :, :, None]).sum(
        dim=3
    )
    expected_value = (
        source_value_candidates * source_weights[:, None, :, :, None]
    ).sum(dim=3)
    assert torch.allclose(key, expected_key, atol=1e-6)
    assert torch.allclose(value, expected_value, atol=1e-6)
    assert projector.last_learned_alignment_key_top1_mean == pytest.approx(0.5)
    assert projector.last_learned_alignment_value_top1_mean == pytest.approx(0.5)


def test_learned_alignment_span_log_prior_falls_back_to_uniform_without_weights():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=8,
        intermediate_dim=8,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_hidden_dim=4,
        learned_alignment_init="anchor",
        learned_alignment_anchor_logit=4.0,
        learned_alignment_prior_mode="span_log_prior",
        learned_alignment_prior_strength=1.0,
    )

    source_key_candidates = torch.randn(1, 1, 1, 3, 2)
    source_value_candidates = torch.randn(1, 1, 1, 3, 2)
    target_key = torch.zeros(1, 1, 1, 2)
    target_value = torch.zeros(1, 1, 1, 2)
    valid_mask = torch.tensor([[[True, True, True]]])

    projector.align_source_kv(
        source_kv_candidates=(source_key_candidates, source_value_candidates),
        target_kv=(target_key, target_value),
        valid_mask=valid_mask,
    )

    assert projector.last_learned_alignment_key_anchor_mean == pytest.approx(1.0 / 3.0)
    assert projector.last_learned_alignment_value_anchor_mean == pytest.approx(
        1.0 / 3.0
    )


def test_learned_alignment_prior_regularization_enters_alignment_loss():
    torch.manual_seed(0)
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=8,
        intermediate_dim=8,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_hidden_dim=4,
        learned_alignment_init="uniform",
        learned_alignment_prior_mode="span_log_prior",
        learned_alignment_prior_strength=1.0,
        learned_alignment_delta_max=0.25,
        learned_alignment_delta_l2_weight=0.1,
        learned_alignment_prior_ce_weight=0.1,
    )
    with torch.no_grad():
        projector.learned_key_alignment_score.bias.fill_(10.0)
        projector.learned_value_alignment_score.bias.fill_(-10.0)

    source_key_candidates = torch.randn(1, 1, 1, 3, 2)
    source_value_candidates = torch.randn(1, 1, 1, 3, 2)
    target_key = torch.zeros(1, 1, 1, 2)
    target_value = torch.zeros(1, 1, 1, 2)
    valid_mask = torch.tensor([[[True, True, True]]])
    source_weights = torch.tensor([[[0.7, 0.2, 0.1]]])

    projector.align_source_kv(
        source_kv_candidates=(source_key_candidates, source_value_candidates),
        target_kv=(target_key, target_value),
        valid_mask=valid_mask,
        source_weights=source_weights,
    )
    aux_loss = projector.alignment_regularization_loss()

    assert aux_loss is not None
    assert aux_loss.item() > 0
    assert projector.last_learned_alignment_delta_l2 <= 0.25 * 0.25 + 1e-6
    assert projector.last_learned_alignment_prior_ce > 0
    assert projector.last_learned_alignment_prior_selected_rate == pytest.approx(1.0)


def test_learned_alignment_grad_aux_loss_targets_loss_reducing_candidate():
    torch.manual_seed(0)
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=8,
        intermediate_dim=8,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_hidden_dim=4,
        learned_alignment_init="uniform",
        learned_alignment_aux_loss_mode="grad_ce",
        learned_alignment_aux_loss_weight=0.1,
        learned_alignment_aux_apply_mode="ambiguous",
        learned_alignment_aux_score_temperature=0.5,
        learned_alignment_aux_score_normalize=False,
    )

    source_key_candidates = torch.tensor(
        [[[[[3.0, 0.0], [-1.0, 0.0], [0.0, 2.0]]]]],
        requires_grad=False,
    )
    source_value_candidates = source_key_candidates.clone()
    target_key = torch.zeros(1, 1, 1, 2)
    target_value = torch.zeros(1, 1, 1, 2)
    valid_mask = torch.tensor([[[True, True, True]]])

    key, value = projector.align_source_kv(
        source_kv_candidates=(source_key_candidates, source_value_candidates),
        target_kv=(target_key, target_value),
        valid_mask=valid_mask,
    )
    task_loss = -(key[..., 0].mean() + value[..., 0].mean())
    aux_loss = projector.compute_learned_alignment_grad_auxiliary_loss(task_loss)

    assert aux_loss is not None
    assert aux_loss.requires_grad
    assert aux_loss.item() > 0
    assert projector.last_learned_alignment_aux_selected_rate == 1.0
    assert projector.last_learned_alignment_aux_target_anchor > 0.9

    aux_loss.backward()

    assert projector.learned_key_alignment_score.weight.grad is not None
    assert projector.learned_value_alignment_score.weight.grad is not None
    assert torch.any(projector.learned_key_alignment_score.weight.grad != 0)
    assert torch.any(projector.learned_value_alignment_score.weight.grad != 0)


def test_learned_alignment_grad_aux_top_r_sparsifies_target():
    torch.manual_seed(0)
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=8,
        intermediate_dim=8,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_hidden_dim=4,
        learned_alignment_init="uniform",
        learned_alignment_aux_loss_mode="grad_ce",
        learned_alignment_aux_loss_weight=0.1,
        learned_alignment_aux_apply_mode="ambiguous",
        learned_alignment_aux_score_temperature=5.0,
        learned_alignment_aux_score_normalize=False,
        learned_alignment_aux_top_r=1,
    )

    source_key_candidates = torch.tensor(
        [[[[[4.0, 0.0], [-1.0, 0.0], [0.0, 2.0]]]]],
        requires_grad=False,
    )
    source_value_candidates = source_key_candidates.clone()
    target_key = torch.zeros(1, 1, 1, 2)
    target_value = torch.zeros(1, 1, 1, 2)
    valid_mask = torch.tensor([[[True, True, True]]])

    key, value = projector.align_source_kv(
        source_kv_candidates=(source_key_candidates, source_value_candidates),
        target_kv=(target_key, target_value),
        valid_mask=valid_mask,
    )
    task_loss = -(key[..., 0].mean() + value[..., 0].mean())
    aux_loss = projector.compute_learned_alignment_grad_auxiliary_loss(task_loss)

    assert aux_loss is not None
    assert projector.last_learned_alignment_aux_target_top1 == pytest.approx(1.0)
    assert projector.last_learned_alignment_aux_target_entropy == pytest.approx(0.0)
    assert projector.last_learned_alignment_aux_target_anchor == pytest.approx(1.0)


def test_learned_alignment_grad_aux_margin_threshold_filters_rows():
    torch.manual_seed(0)
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=8,
        intermediate_dim=8,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_hidden_dim=4,
        learned_alignment_init="uniform",
        learned_alignment_aux_loss_mode="grad_ce",
        learned_alignment_aux_loss_weight=0.1,
        learned_alignment_aux_apply_mode="ambiguous",
        learned_alignment_aux_score_temperature=1.0,
        learned_alignment_aux_score_normalize=False,
        learned_alignment_aux_score_margin_threshold=0.5,
    )

    source_key_candidates = torch.tensor(
        [
            [
                [
                    [[4.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
                    [[1.00, 0.0], [0.95, 0.0], [0.90, 0.0]],
                ]
            ]
        ],
        requires_grad=False,
    )
    source_value_candidates = source_key_candidates.clone()
    target_key = torch.zeros(1, 1, 2, 2)
    target_value = torch.zeros(1, 1, 2, 2)
    valid_mask = torch.tensor([[[True, True, True], [True, True, True]]])

    key, value = projector.align_source_kv(
        source_kv_candidates=(source_key_candidates, source_value_candidates),
        target_kv=(target_key, target_value),
        valid_mask=valid_mask,
    )
    task_loss = -(key[..., 0].sum() + value[..., 0].sum())
    aux_loss = projector.compute_learned_alignment_grad_auxiliary_loss(task_loss)

    assert aux_loss is not None
    assert aux_loss.requires_grad
    assert projector.last_learned_alignment_aux_selected_rate == pytest.approx(0.5)
    assert projector.last_learned_alignment_aux_target_anchor > 0.9


def test_learned_alignment_forced_rank_selects_requested_candidate():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=8,
        intermediate_dim=8,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_hidden_dim=4,
        learned_alignment_init="uniform",
    )

    source_key_candidates = torch.tensor(
        [[[[[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]]]],
    )
    source_value_candidates = source_key_candidates.clone()
    target_key = torch.zeros(1, 1, 1, 2)
    target_value = torch.zeros(1, 1, 1, 2)
    valid_mask = torch.tensor([[[True, True, True]]])

    projector.set_learned_alignment_forced_rank(1)
    key, value = projector.align_source_kv(
        source_kv_candidates=(source_key_candidates, source_value_candidates),
        target_kv=(target_key, target_value),
        valid_mask=valid_mask,
    )

    assert key[0, 0, 0, 0].item() == pytest.approx(3.0)
    assert value[0, 0, 0, 0].item() == pytest.approx(3.0)


def test_learned_alignment_replay_ce_uses_batch_level_target():
    torch.manual_seed(0)
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=8,
        intermediate_dim=8,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_hidden_dim=4,
        learned_alignment_init="uniform",
        learned_alignment_aux_loss_mode="replay_ce",
        learned_alignment_aux_loss_weight=0.1,
        learned_alignment_aux_apply_mode="ambiguous",
    )

    source_key_candidates = torch.tensor(
        [[[[[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]]]],
    )
    source_value_candidates = source_key_candidates.clone()
    target_key = torch.zeros(1, 1, 1, 2)
    target_value = torch.zeros(1, 1, 1, 2)
    valid_mask = torch.tensor([[[True, True, True]]])

    projector.set_learned_alignment_replay_target(torch.tensor([0.1, 0.8, 0.1]))
    projector.align_source_kv(
        source_kv_candidates=(source_key_candidates, source_value_candidates),
        target_kv=(target_key, target_value),
        valid_mask=valid_mask,
    )
    aux_loss = projector.alignment_regularization_loss()

    assert aux_loss is not None
    assert aux_loss.requires_grad
    assert projector.last_learned_alignment_aux_selected_rate == pytest.approx(1.0)
    assert projector.last_learned_alignment_aux_target_top1 == pytest.approx(0.8)
    assert projector.last_learned_alignment_aux_target_anchor == pytest.approx(0.1)

    aux_loss.backward()

    assert projector.learned_key_alignment_score.weight.grad is not None
    assert projector.learned_value_alignment_score.weight.grad is not None
    assert torch.any(projector.learned_key_alignment_score.weight.grad != 0)
    assert torch.any(projector.learned_value_alignment_score.weight.grad != 0)


def test_learned_alignment_margin_rank_uses_candidate_utility():
    torch.manual_seed(0)
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=8,
        intermediate_dim=8,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_hidden_dim=4,
        learned_alignment_init="uniform",
        learned_alignment_aux_loss_mode="grad_ce_margin_rank",
        learned_alignment_aux_loss_weight=0.0,
        learned_alignment_margin_rank_loss_weight=0.1,
        learned_alignment_margin_rank_threshold=0.25,
    )

    source_key_candidates = torch.tensor(
        [[[[[1.0, 0.0], [3.0, 0.0], [-1.0, 0.0]]]]],
    )
    source_value_candidates = source_key_candidates.clone()
    target_key = torch.zeros(1, 1, 1, 2)
    target_value = torch.zeros(1, 1, 1, 2)
    valid_mask = torch.tensor([[[True, True, True]]])

    projector.set_learned_alignment_replay_utility(torch.tensor([0.0, 2.0, -1.0]))
    projector.align_source_kv(
        source_kv_candidates=(source_key_candidates, source_value_candidates),
        target_kv=(target_key, target_value),
        valid_mask=valid_mask,
    )
    aux_loss = projector.alignment_regularization_loss()

    assert aux_loss is not None
    assert aux_loss.requires_grad
    assert projector.last_learned_alignment_margin_rank_selected_rate == pytest.approx(
        1.0
    )
    assert projector.last_learned_alignment_margin_rank_utility_top1 == pytest.approx(
        1.0
    )
    assert projector.last_learned_alignment_margin_rank_pair_count > 0

    aux_loss.backward()

    assert projector.learned_key_alignment_score.weight.grad is not None
    assert projector.learned_value_alignment_score.weight.grad is not None
    assert torch.any(projector.learned_key_alignment_score.weight.grad != 0)
    assert torch.any(projector.learned_value_alignment_score.weight.grad != 0)


def test_learned_alignment_margin_rank_can_use_batch_mean_scope():
    torch.manual_seed(0)
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=8,
        intermediate_dim=8,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_hidden_dim=4,
        learned_alignment_init="uniform",
        learned_alignment_aux_loss_mode="grad_ce_margin_rank",
        learned_alignment_aux_loss_weight=0.0,
        learned_alignment_margin_rank_loss_weight=0.1,
        learned_alignment_margin_rank_threshold=0.25,
        learned_alignment_margin_rank_scope="batch_mean",
    )

    source_key_candidates = torch.tensor(
        [
            [
                [
                    [[1.0, 0.0], [3.0, 0.0], [-1.0, 0.0]],
                    [[2.0, 0.0], [4.0, 0.0], [-2.0, 0.0]],
                ]
            ]
        ],
    )
    source_value_candidates = source_key_candidates.clone()
    target_key = torch.zeros(1, 1, 2, 2)
    target_value = torch.zeros(1, 1, 2, 2)
    valid_mask = torch.tensor([[[True, True, True], [True, True, True]]])

    projector.set_learned_alignment_replay_utility(torch.tensor([0.0, 2.0, -1.0]))
    projector.align_source_kv(
        source_kv_candidates=(source_key_candidates, source_value_candidates),
        target_kv=(target_key, target_value),
        valid_mask=valid_mask,
    )
    aux_loss = projector.alignment_regularization_loss()

    assert aux_loss is not None
    assert aux_loss.requires_grad
    assert projector.last_learned_alignment_margin_rank_pair_count > 0
    assert projector.last_learned_alignment_margin_rank_utility_top1 == pytest.approx(
        1.0
    )

    aux_loss.backward()

    assert projector.learned_key_alignment_score.weight.grad is not None
    assert torch.any(projector.learned_key_alignment_score.weight.grad != 0)


def test_learned_alignment_margin_rank_ignores_unscored_padded_candidates():
    torch.manual_seed(0)
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=8,
        intermediate_dim=8,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_hidden_dim=4,
        learned_alignment_init="uniform",
        learned_alignment_aux_loss_mode="grad_ce_margin_rank",
        learned_alignment_aux_loss_weight=0.0,
        learned_alignment_margin_rank_loss_weight=0.1,
        learned_alignment_margin_rank_threshold=0.1,
        learned_alignment_margin_rank_scope="batch_mean",
    )

    source_key_candidates = torch.randn(1, 1, 2, 5, 2)
    source_value_candidates = source_key_candidates.clone()
    target_key = torch.zeros(1, 1, 2, 2)
    target_value = torch.zeros(1, 1, 2, 2)
    valid_mask = torch.ones(1, 2, 5, dtype=torch.bool)

    # Only ranks 0 and 1 are scored. Padded ranks 2-4 would look better than
    # negative margins if padding were treated as real utility.
    projector.set_learned_alignment_replay_utility(torch.tensor([-2.0, -1.0]))
    projector.align_source_kv(
        source_kv_candidates=(source_key_candidates, source_value_candidates),
        target_kv=(target_key, target_value),
        valid_mask=valid_mask,
    )
    aux_loss = projector.alignment_regularization_loss()

    assert aux_loss is not None
    assert projector.last_learned_alignment_margin_rank_pair_count == pytest.approx(1.0)
    assert projector.last_learned_alignment_margin_rank_utility_top1 == pytest.approx(
        1.0
    )


def test_candidate_replay_answer_token_score_uses_option_label_only():
    logits = torch.zeros(1, 5, 8)
    labels = torch.tensor([[-100, 2, 3, 6, 1]])

    # Shifted labels are [2, 3, 6, 1], so logits[:, 2] predicts answer token 6.
    logits[0, 2, 6] = 5.0
    logits[0, 2, 1] = -5.0
    # Make a non-answer position very bad; answer_token_ce should ignore it.
    logits[0, 0, 4] = -10.0

    loss, metrics = _candidate_replay_score_loss_from_logits(
        logits=logits,
        labels=labels,
        task_loss=torch.tensor(123.0),
        config={
            "score_mode": "answer_token_ce",
            "option_token_ids": [4, 5, 6, 7],
            "min_score_positions": 1,
        },
    )

    assert metrics["candidate_replay/score_positions"] == pytest.approx(1.0)
    assert metrics["candidate_replay/score_task_loss_fallback"] == pytest.approx(0.0)
    assert loss.item() < 0.1


def test_candidate_replay_answer_margin_scores_correct_vs_wrong_options():
    logits = torch.zeros(1, 5, 8)
    labels = torch.tensor([[-100, 2, 3, 6, 1]])

    # Shifted labels are [2, 3, 6, 1], so logits[:, 2] predicts answer token 6.
    logits[0, 2, 6] = 5.0
    logits[0, 2, 4] = 1.0
    logits[0, 2, 5] = 0.5
    logits[0, 2, 7] = -0.5

    utility, metrics = _candidate_replay_answer_margin_from_logits(
        logits=logits,
        labels=labels,
        task_loss=torch.tensor(123.0),
        config={
            "score_mode": "answer_margin",
            "option_token_ids": [4, 5, 6, 7],
            "min_score_positions": 1,
        },
    )

    assert metrics["candidate_replay/score_positions"] == pytest.approx(1.0)
    assert metrics["candidate_replay/score_task_loss_fallback"] == pytest.approx(0.0)
    assert utility.item() > 3.0


def test_candidate_replay_answer_score_falls_back_to_suffix_ce():
    logits = torch.zeros(1, 6, 8)
    labels = torch.tensor([[-100, 2, 3, 4, 5, 6]])

    # No ABCD option id appears in shifted labels. The helper should score the
    # last two supervised shifted labels: 5 and 6.
    logits[0, 3, 5] = 4.0
    logits[0, 4, 6] = 4.0
    logits[0, 0, 2] = -10.0

    loss, metrics = _candidate_replay_score_loss_from_logits(
        logits=logits,
        labels=labels,
        task_loss=torch.tensor(123.0),
        config={
            "score_mode": "answer_token_ce",
            "option_token_ids": [10, 11, 12, 13],
            "min_score_positions": 1,
            "fallback_score_mode": "suffix_ce",
            "fallback_suffix_tokens": 2,
        },
    )

    assert metrics["candidate_replay/score_positions"] == pytest.approx(2.0)
    assert metrics["candidate_replay/score_fallback_used"] == pytest.approx(1.0)
    assert metrics["candidate_replay/score_task_loss_fallback"] == pytest.approx(0.0)
    assert loss.item() < 0.2


def test_learned_alignment_injection_gate_starts_neutral_and_trains():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=2,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_injection_gate_mode="token_mlp",
        learned_alignment_injection_init_logit=0.0,
    )

    key_hidden = torch.ones(1, 3, 4, requires_grad=True)
    value_hidden = torch.ones(1, 3, 4, requires_grad=True)
    key_gate, value_gate = projector._compute_learned_alignment_injection_gate(
        key_hidden=key_hidden,
        value_hidden=value_hidden,
        target_shape=(1, 2, 3, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert torch.allclose(key_gate, torch.full_like(key_gate, 0.5), atol=1e-6)
    assert torch.allclose(value_gate, torch.full_like(value_gate, 0.5), atol=1e-6)
    assert projector.last_learned_alignment_key_injection == 0.5
    assert projector.last_learned_alignment_value_injection == 0.5

    with torch.no_grad():
        projector.learned_key_injection_gate_bias.fill_(2.0)
        projector.learned_value_injection_gate_bias.fill_(-2.0)

    key_gate_after, value_gate_after = (
        projector._compute_learned_alignment_injection_gate(
            key_hidden=key_hidden,
            value_hidden=value_hidden,
            target_shape=(1, 2, 3, 2),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    )

    assert torch.all(key_gate_after > 0.5)
    assert torch.all(value_gate_after < 0.5)

    loss = key_gate_after.mean() + value_gate_after.mean()
    loss.backward()

    assert projector.learned_key_injection_gate_bias.grad is not None
    assert projector.learned_value_injection_gate_bias.grad is not None
    assert torch.any(projector.learned_key_injection_gate_bias.grad != 0)
    assert torch.any(projector.learned_value_injection_gate_bias.grad != 0)


def test_learned_alignment_transfer_gate_reduces_uncertain_router_rows():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=2,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_transfer_gate_mode="router_quality",
        learned_alignment_transfer_gate_floor=0.5,
        learned_alignment_transfer_gate_entropy_threshold=0.5,
        learned_alignment_transfer_gate_margin_threshold=0.5,
        learned_alignment_transfer_gate_temperature=0.05,
        learned_alignment_transfer_gate_min_valid=2,
    )
    valid_mask = torch.tensor([[[True, True, True], [True, False, False]]])
    weights = torch.tensor([[[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], [1.0, 0.0, 0.0]]])

    gate, entropy, margin, top1, selected = (
        projector._learned_alignment_transfer_gate_from_weights(
            weights=weights,
            valid_mask=valid_mask,
            target_shape=(1, 2, 2, 2),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    )

    assert gate.shape == (1, 2, 2, 1)
    assert gate[0, 0, 0, 0].item() < 0.6
    assert gate[0, 0, 1, 0].item() == pytest.approx(1.0)
    assert entropy[0, 0].item() == pytest.approx(1.0)
    assert margin[0, 0].item() == pytest.approx(0.0)
    assert top1[0, 0].item() == pytest.approx(1.0 / 3.0)
    assert selected.tolist() == [[True, False]]


def test_learned_alignment_transfer_gate_keeps_confident_router_rows():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_transfer_gate_mode="router_quality",
        learned_alignment_transfer_gate_floor=0.5,
        learned_alignment_transfer_gate_entropy_threshold=0.9,
        learned_alignment_transfer_gate_margin_threshold=0.2,
        learned_alignment_transfer_gate_temperature=0.05,
        learned_alignment_transfer_gate_min_valid=2,
    )
    valid_mask = torch.tensor([[[True, True, True]]])
    weights = torch.tensor([[[0.9, 0.05, 0.05]]])

    gate, _, margin, top1, selected = (
        projector._learned_alignment_transfer_gate_from_weights(
            weights=weights,
            valid_mask=valid_mask,
            target_shape=(1, 1, 1, 2),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    )

    assert gate[0, 0, 0, 0].item() > 0.98
    assert margin[0, 0].item() == pytest.approx(0.85)
    assert top1[0, 0].item() == pytest.approx(0.9)
    assert selected.tolist() == [[True]]


def test_learned_alignment_transfer_gate_multiplies_forward_residual():
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        learned_alignment_mode="kv_router",
        learned_alignment_transfer_gate_mode="router_quality",
        learned_alignment_transfer_gate_floor=0.5,
        learned_alignment_transfer_gate_entropy_threshold=0.5,
        learned_alignment_transfer_gate_margin_threshold=0.5,
        learned_alignment_transfer_gate_temperature=0.05,
        learned_alignment_transfer_gate_min_valid=2,
    )
    projector.eval()
    with torch.no_grad():
        projector.key_gate_logit.fill_(1.0)
        projector.value_gate_logit.fill_(1.0)
    source_key = torch.randn(1, 1, 1, 2)
    source_value = torch.randn(1, 1, 1, 2)
    target_key = torch.zeros(1, 1, 1, 2)
    target_value = torch.zeros(1, 1, 1, 2)

    projector._last_learned_alignment_key_transfer_gate = torch.full((1, 1, 1, 1), 0.5)
    projector._last_learned_alignment_value_transfer_gate = torch.full(
        (1, 1, 1, 1),
        0.25,
    )
    out_key_gated, out_value_gated = projector(
        (source_key, source_value),
        (target_key, target_value),
    )

    projector._last_learned_alignment_key_transfer_gate = torch.ones(1, 1, 1, 1)
    projector._last_learned_alignment_value_transfer_gate = torch.ones(1, 1, 1, 1)
    out_key_full, out_value_full = projector(
        (source_key, source_value),
        (target_key, target_value),
    )

    assert torch.allclose(out_key_gated, out_key_full * 0.5, atol=1e-6)
    assert torch.allclose(out_value_gated, out_value_full * 0.25, atol=1e-6)


def test_rosetta_candidate_gather_preserves_topk_and_valid_mask():
    source_key_cache = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1)
    source_value_cache = source_key_cache + 10.0
    source_indices = torch.tensor([[[2, 0, -1], [3, 9, 1]]])

    key, value, valid = RosettaModel._source_kv_candidates_from_indices(
        source_key_cache=source_key_cache,
        source_value_cache=source_value_cache,
        source_indices=source_indices,
    )

    assert key.shape == (1, 1, 2, 3, 1)
    assert value.shape == (1, 1, 2, 3, 1)
    assert valid.tolist() == [[[True, True, False], [True, False, True]]]
    assert key[0, 0, 0, :, 0].tolist() == [2.0, 0.0, 0.0]
    assert value[0, 0, 1, :, 0].tolist() == [13.0, 0.0, 11.0]
