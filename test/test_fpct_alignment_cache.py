from types import SimpleNamespace

import torch

from rosetta.train.dataset_adapters import AlignedChatDataset


def test_frozen_fpct_alignment_cache_is_cloned_and_validated(tmp_path):
    item = {
        "input_ids": [[1, 2], [3, 4]],
        "labels": [-100, 2],
        "kv_cache_index": torch.tensor([[1, 0], [1, 1]]),
        "soft_alignment": {
            "source_indices": torch.tensor([[0, -1], [1, -1]]),
            "source_weights": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        },
    }
    path = tmp_path / "cache.pt"
    torch.save({"schema_version": 1, "alignment_sanitizer": "certified_slot0_v1", "top_k": 2, "items": [item]}, path)
    dataset = AlignedChatDataset([[]], SimpleNamespace(strategy=None), soft_alignment_top_k=2, fpct_alignment_sanitizer="certified_slot0_v1", fpct_alignment_cache_path=str(path))
    first = dataset[0]; second = dataset[0]
    first["soft_alignment"]["source_weights"][0, 0] = 0
    assert second["soft_alignment"]["source_weights"][0, 0] == 1
