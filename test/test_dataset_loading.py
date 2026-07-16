from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset, load_dataset as hf_load_dataset

from rosetta.utils import dataset_loading


def _use_test_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = tmp_path / "cache"

    def load_with_test_cache(*args, **kwargs):
        kwargs["cache_dir"] = str(cache_dir)
        return hf_load_dataset(*args, **kwargs)

    monkeypatch.setattr(dataset_loading, "load_dataset", load_with_test_cache)


def test_data_root_prefers_explicit_path_then_environment(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment_root = tmp_path / "environment"
    explicit_root = tmp_path / "explicit"
    monkeypatch.setenv("C2C_DATA_ROOT", str(environment_root))

    assert dataset_loading.data_root() == environment_root
    assert dataset_loading.data_root(explicit_root) == explicit_root


def test_mmlu_local_loader_selects_only_requested_split(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_test_cache(tmp_path, monkeypatch)
    data_directory = tmp_path / "c2c" / "mmlu" / "all"
    data_directory.mkdir(parents=True)
    pq.write_table(
        pa.table({"question": ["aux"], "answer": [0]}),
        data_directory / "auxiliary_train-00000-of-00001.parquet",
    )
    pq.write_table(
        pa.table({"question": ["test"], "answer": [1]}),
        data_directory / "test-00000-of-00001.parquet",
    )

    dataset = dataset_loading.load_c2c_dataset(
        "cais/mmlu",
        config_name="all",
        split="auxiliary_train",
        data_root_path=tmp_path / "c2c",
    )

    assert len(dataset) == 1
    assert dataset[0]["question"] == "aux"


def test_redux_local_loader_assigns_generic_arrow_to_requested_split(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_test_cache(tmp_path, monkeypatch)
    data_directory = tmp_path / "c2c" / "mmlu-redux-2.0" / "abstract_algebra"
    Dataset.from_dict({"question": ["q"], "answer": [0]}).save_to_disk(
        str(data_directory)
    )

    dataset = dataset_loading.load_c2c_dataset(
        "edinburgh-dawg/mmlu-redux-2.0",
        config_name="abstract_algebra",
        split="test",
        data_root_path=tmp_path / "c2c",
    )

    assert len(dataset) == 1
    assert dataset[0]["question"] == "q"


def test_openhermes_local_loader_reads_json(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_test_cache(tmp_path, monkeypatch)
    data_directory = tmp_path / "c2c" / "OpenHermes-2.5"
    data_directory.mkdir(parents=True)
    (data_directory / "openhermes2_5.json").write_text(
        json.dumps([{"conversations": [{"from": "human", "value": "hi"}]}]),
        encoding="utf-8",
    )

    dataset = dataset_loading.load_c2c_dataset(
        "teknium/OpenHermes-2.5",
        split="train",
        data_root_path=tmp_path / "c2c",
    )

    assert len(dataset) == 1
    assert dataset[0]["conversations"][0]["value"] == "hi"


def test_missing_local_dataset_falls_back_to_huggingface(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    sentinel = object()

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(dataset_loading, "load_dataset", fake_load_dataset)

    result = dataset_loading.load_c2c_dataset(
        "ceval/ceval-exam",
        config_name="accountant",
        split="test",
        data_root_path=tmp_path / "missing",
    )

    assert result is sentinel
    assert calls == [(("ceval/ceval-exam", "accountant"), {"split": "test"})]


def test_existing_but_incomplete_local_dataset_fails_loudly(tmp_path) -> None:
    (tmp_path / "c2c" / "mmlu").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="配置目录不存在"):
        dataset_loading.load_c2c_dataset(
            "cais/mmlu",
            config_name="all",
            split="auxiliary_train",
            data_root_path=tmp_path / "c2c",
        )
