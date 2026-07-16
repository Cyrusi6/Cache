"""Resolve C2C datasets from the unified local data root before Hugging Face."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_HOST_DATA_ROOT = Path("/home/lijunsi/projects/KVcache/datasets/c2c")
DEFAULT_CONTAINER_DATA_ROOT = Path("/datasets/c2c")

DATASET_DIRECTORIES = {
    "teknium/openhermes-2.5": "OpenHermes-2.5",
    "openhermes-2.5": "OpenHermes-2.5",
    "cais/mmlu": "mmlu",
    "mmlu": "mmlu",
    "edinburgh-dawg/mmlu-redux-2.0": "mmlu-redux-2.0",
    "mmlu-redux": "mmlu-redux-2.0",
    "mmlu-redux-2.0": "mmlu-redux-2.0",
    "thudm/longbench": "LongBench",
    "xnhyacinth/longbench": "LongBench",
    "longbench": "LongBench",
    "openbookqa": "openbookqa",
    "allenai/openbookqa": "openbookqa",
    "allenai/ai2_arc": "ai2_arc",
    "ai2-arc": "ai2_arc",
    "openai/gsm8k": "gsm8k",
    "gsm8k": "gsm8k",
    "ceval/ceval-exam": "ceval-exam",
    "ceval": "ceval-exam",
}


def data_root(data_root: str | os.PathLike[str] | None = None) -> Path:
    """Return the configured unified data root without requiring it to exist."""
    if data_root is not None:
        return Path(data_root).expanduser()
    configured = os.environ.get("C2C_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    if DEFAULT_CONTAINER_DATA_ROOT.is_dir():
        return DEFAULT_CONTAINER_DATA_ROOT
    return DEFAULT_HOST_DATA_ROOT


def local_dataset_path(
    dataset_name: str, data_root_path: str | os.PathLike[str] | None = None
) -> Path | None:
    """Return a readable local dataset directory for a known C2C dataset."""
    directory_name = DATASET_DIRECTORIES.get(dataset_name.lower())
    if directory_name is None:
        return None
    candidate = data_root(data_root_path) / directory_name
    return candidate if candidate.is_dir() else None


def _local_data_directory(dataset_path: Path, config_name: str | None) -> Path:
    if config_name is None:
        return dataset_path
    config_path = dataset_path / config_name
    if not config_path.is_dir():
        raise FileNotFoundError(f"本地数据集配置目录不存在：{config_path}")
    return config_path


def _matching_files(data_directory: Path, split: str, suffix: str) -> list[Path]:
    patterns = (
        f"{split}-*.{suffix}",
        f"{split}.{suffix}",
        f"*{split}*.{suffix}",
    )
    for pattern in patterns:
        files = sorted(data_directory.glob(pattern))
        if files:
            return files
    return sorted(data_directory.glob(f"*.{suffix}"))


def _load_local_dataset(dataset_path: Path, config_name: str | None, split: str) -> Any:
    data_directory = _local_data_directory(dataset_path, config_name)
    builders = (
        ("parquet", "parquet"),
        ("arrow", "arrow"),
        ("json", "json"),
        ("json", "jsonl"),
        ("csv", "csv"),
    )
    for builder, suffix in builders:
        files = _matching_files(data_directory, split, suffix)
        if files:
            print(
                f"[c2c-data] local={data_directory} split={split}",
                flush=True,
            )
            return load_dataset(
                builder,
                data_files={split: [str(path) for path in files]},
                split=split,
            )
    raise FileNotFoundError(
        f"本地数据目录中没有可加载的 {split} 文件：{data_directory}"
    )


def load_c2c_dataset(
    dataset_name: str,
    *,
    config_name: str | None = None,
    split: str,
    data_root_path: str | os.PathLike[str] | None = None,
) -> Any:
    """Load one split from the unified local root, falling back to Hugging Face."""
    dataset_path = local_dataset_path(dataset_name, data_root_path)
    if dataset_path is not None:
        return _load_local_dataset(dataset_path, config_name, split)

    print(
        f"[c2c-data] huggingface={dataset_name} config={config_name} split={split}",
        flush=True,
    )
    if config_name is None:
        return load_dataset(dataset_name, split=split)
    return load_dataset(dataset_name, config_name, split=split)
