from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from script.train.SFT_train import (
    _create_dataset_split,
    _load_dataset_split_indices,
    _write_dataset_split_indices,
)


def test_frozen_split_manifest_roundtrip_preserves_order_and_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "split.json"
    train_indices = [4, 0, 5, 2]
    eval_indices = [3, 1]

    _write_dataset_split_indices(
        path,
        train_indices,
        eval_indices,
        dataset_size=6,
        seed=42,
        split_mode="seeded",
    )

    assert _load_dataset_split_indices(path, 6, 4, 2) == (
        train_indices,
        eval_indices,
    )
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "dataset_size": 6,
        "seed": 42,
        "split_mode": "seeded",
        "train_indices": train_indices,
        "eval_indices": eval_indices,
    }

    train, evaluation, mode = _create_dataset_split(
        list("abcdef"),
        4,
        2,
        seed=999,
        split_indices_path=path,
    )
    assert mode == "frozen_indices"
    assert train.indices == train_indices
    assert evaluation.indices == eval_indices


def test_frozen_split_manifest_refuses_different_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    _write_dataset_split_indices(
        path,
        [0, 1, 2],
        [3],
        dataset_size=4,
        seed=42,
        split_mode="legacy_global_rng",
    )

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        _write_dataset_split_indices(
            path,
            [1, 0, 2],
            [3],
            dataset_size=4,
            seed=42,
            split_mode="legacy_global_rng",
        )


@pytest.mark.parametrize(
    ("train_indices", "eval_indices", "message"),
    [
        ([0, 2, 4], [1, 3], "has lengths 3/2; expected 4/2"),
        ([0, 2, 2, 5], [1, 3], "every dataset index exactly once"),
        ([0, 2, 4, 6], [1, 3], "every dataset index exactly once"),
    ],
    ids=["wrong-length", "duplicate", "out-of-range"],
)
def test_frozen_split_manifest_rejects_invalid_indices(
    tmp_path: Path,
    train_indices: list[int],
    eval_indices: list[int],
    message: str,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "train_indices": train_indices,
                "eval_indices": eval_indices,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        _load_dataset_split_indices(path, dataset_size=6, train_size=4, eval_size=2)


def test_frozen_split_manifest_rejects_bad_declared_size_and_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid-metadata.json"
    payload = {
        "dataset_size": 7,
        "train_indices": [0, 1, 2, 3],
        "eval_indices": [4, 5],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="declares dataset_size=7"):
        _load_dataset_split_indices(path, dataset_size=6, train_size=4, eval_size=2)

    payload["dataset_size"] = 6
    payload["indices_sha256"] = {"train": "0" * 64}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid train index SHA256"):
        _load_dataset_split_indices(path, dataset_size=6, train_size=4, eval_size=2)


def test_default_seeded_split_matches_previous_explicit_generator_behavior() -> None:
    dataset = list(range(10))
    seed = 42
    expected_train, expected_eval = torch.utils.data.random_split(
        dataset,
        [8, 2],
        generator=torch.Generator().manual_seed(seed),
    )
    torch.manual_seed(1234)
    global_rng_before = torch.random.get_rng_state().clone()

    train, evaluation, mode = _create_dataset_split(
        dataset,
        8,
        2,
        seed=seed,
    )

    assert mode == "seeded"
    assert train.indices == expected_train.indices
    assert evaluation.indices == expected_eval.indices
    assert torch.equal(torch.random.get_rng_state(), global_rng_before)


def test_legacy_split_matches_process_global_rng_behavior() -> None:
    dataset = list(range(10))
    torch.manual_seed(7)
    expected_train, expected_eval = torch.utils.data.random_split(dataset, [8, 2])
    torch.manual_seed(7)

    train, evaluation, mode = _create_dataset_split(
        dataset,
        8,
        2,
        seed=999,
        split_mode="legacy_global_rng",
    )

    assert mode == "legacy_global_rng"
    assert train.indices == expected_train.indices
    assert evaluation.indices == expected_eval.indices
