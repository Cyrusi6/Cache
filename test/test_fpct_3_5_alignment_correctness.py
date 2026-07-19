from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "script/analysis/fpct_3_5_alignment_correctness.py"
SPEC = importlib.util.spec_from_file_location("fpct_3_5_alignment_correctness", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
correctness = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = correctness
SPEC.loader.exec_module(correctness)


def snapshot(*, text="abc", ids=(10, 11, 12), offsets=((0, 1), (1, 2), (2, 3))):
    return correctness.IdentitySnapshot(
        rendered_text=text,
        input_ids=ids,
        offsets=offsets,
        content_spans=((0, 3),),
        message_ranges=((0, 3),),
    )


def test_identical_tokenizer_duplicate_offsets_force_identity_m1() -> None:
    details = snapshot(offsets=((0, 1), (1, 2), (1, 2)))
    alignment = correctness.exact_identity_alignment(details)
    assert alignment["source_indices"] == [[0, -1, -1, -1], [1, -1, -1, -1], [2, -1, -1, -1]]
    assert alignment["source_weights"] == [[1.0, 0.0, 0.0, 0.0]] * 3
    assert alignment["fallback_mask"] == [False, False, False]


def test_identical_tokenizer_ordinary_offsets_force_identity_m1() -> None:
    alignment = correctness.exact_identity_alignment(snapshot())
    assert all(sum(weight > 0 for weight in row) == 1 for row in alignment["source_weights"])
    assert all(row[0] == index for index, row in enumerate(alignment["source_indices"]))


def test_genuine_disjoint_partition_is_certified() -> None:
    result = correctness.certify_one_to_many(
        receiver_interval=(0, 4),
        other_receiver_intervals=((4, 6),),
        candidate_indices=(7, 8),
        candidate_intervals=((0, 2), (2, 4)),
    )
    assert result.certified
    assert result.category == "certified_one_to_many"


def test_duplicate_source_offsets_are_uncertified() -> None:
    result = correctness.certify_one_to_many(
        receiver_interval=(0, 2),
        other_receiver_intervals=(),
        candidate_indices=(7, 8),
        candidate_intervals=((0, 2), (0, 2)),
    )
    assert not result.certified
    assert result.category == "exact_duplicate_source_offsets"


def test_overlapping_receiver_offsets_are_uncertified() -> None:
    result = correctness.certify_one_to_many(
        receiver_interval=(0, 2),
        other_receiver_intervals=((1, 3),),
        candidate_indices=(7, 8),
        candidate_intervals=((0, 1), (1, 2)),
    )
    assert not result.certified
    assert result.category == "duplicate_or_overlap_receiver_offsets"


@pytest.mark.parametrize(
    "receiver,sender,error",
    [
        (snapshot(text="abc"), snapshot(text="abd"), "rendered_text_difference"),
        (snapshot(ids=(10, 11, 12)), snapshot(ids=(10, 99, 12)), "token_id_difference"),
        (snapshot(), snapshot(offsets=((0, 1), (1, 3), (2, 3))), "offset_difference"),
    ],
)
def test_exact_identity_mismatch_is_hard_error(receiver, sender, error) -> None:
    fingerprint = {"sha256": "same"}
    with pytest.raises(correctness.AlignmentCorrectnessError, match=error):
        correctness.assert_exact_runtime_identity(receiver, sender, fingerprint, fingerprint)


def test_runtime_fingerprint_mismatch_is_hard_error() -> None:
    with pytest.raises(correctness.AlignmentCorrectnessError, match="tokenizer_or_pair_path_mixup"):
        correctness.assert_exact_runtime_identity(
            snapshot(), snapshot(), {"sha256": "left"}, {"sha256": "right"}
        )


def test_sanitizer_is_arm_independent_on_uncertified_row() -> None:
    expected = ([4, -1, -1, -1], [1.0, 0.0, 0.0, 0.0])
    for _operator in ("c_pre", "c_post", "f"):
        assert correctness.sanitize_candidate_row(
            [4, 5, -1, -1], [0.5, 0.5, 0.0, 0.0], certified=False
        ) == expected


def test_certified_row_is_preserved() -> None:
    indices = [4, 5, -1, -1]
    weights = [0.5, 0.5, 0.0, 0.0]
    assert correctness.sanitize_candidate_row(indices, weights, certified=True) == (
        indices,
        weights,
    )


def test_identity_rows_create_no_extra_slots() -> None:
    alignment = correctness.exact_identity_alignment(snapshot())
    extra = sum(
        max(sum(weight > 0 for weight in row) - 1, 0)
        for row, eligible in zip(alignment["source_weights"], alignment["message_mask"])
        if eligible
    )
    assert alignment["extra_slots"] == 0
    assert extra == 0


def test_script_has_no_model_or_gpu_execution() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "AutoModel" not in source
    assert ".cuda(" not in source
    assert "torch.cuda" not in source
