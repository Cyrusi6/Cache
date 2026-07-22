from __future__ import annotations

import json
from pathlib import Path

from script.analysis.fpct_gpu_r2m_finalize import finalize


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_r2m_finalizer_emits_r2m_go_only_after_all_frozen_gates(tmp_path: Path) -> None:
    original = tmp_path / "original.json"
    semantic = tmp_path / "semantic.json"
    active = tmp_path / "active.json"
    output = tmp_path / "final.json"
    write(
        original,
        {
            "status": "GO",
            "checks": {f"check_{index}": True for index in range(23)},
            "accuracy_or_correctness_accessed": False,
        },
    )
    write(
        semantic,
        {
            "status": "GO",
            "checks": {f"semantic_{index}": True for index in range(6)},
            "accuracy_or_correctness_accessed": False,
        },
    )
    write(
        active,
        {
            "canaries": {
                "checkpoint_native": {"qualified": True},
                "forced_on": {"qualified": True},
            },
            "accuracy_or_correctness_accessed": False,
        },
    )
    payload = finalize(original, semantic, active, output)
    assert payload["classification"] == "R2M_IMMUTABLE_GO"
    assert payload["training_authorized"] is True
