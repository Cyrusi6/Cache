from __future__ import annotations

import json
from pathlib import Path

import pytest

from script.experiment import fpct_gpu_r2_controller as controller


def write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value) + "\n")
    return path


def test_r2_controller_is_sequential_and_single_release(tmp_path: Path) -> None:
    lock = write_json(
        tmp_path / "lock.json",
        {"status": "PRE_OUTPUT_LOCKED_R2", "run_uid": "r2-test"},
    )
    root = tmp_path / "state"
    state = controller.initialize(root, lock)
    assert state["stage"] == "R2_RUN_LOCKED"
    gate = write_json(tmp_path / "gate.json", {"status": "GO"})
    controller.transition(root, "R2_GPU_NUMERICAL_GO", gate)
    controller.transition(root, "R2_PRETRAINED_GO", gate)
    controller.transition(root, "MATCHED_SMOKE_GO", gate)
    controller.transition(root, "FORMAL_TRAINING_SUBMITTED")
    with pytest.raises(controller.R2StateError, match="12 matched"):
        controller.transition(root, "FORMAL_TRAINING_COMPLETE")


def test_r2_controller_terminal_is_irreversible(tmp_path: Path) -> None:
    lock = write_json(
        tmp_path / "lock.json",
        {"status": "PRE_OUTPUT_LOCKED_R2", "run_uid": "r2-test"},
    )
    root = tmp_path / "state"
    controller.initialize(root, lock)
    state = controller.transition(root, "GPU_ENGINEERING_BLOCKED_R2")
    assert state["stage"] == "TERMINAL"
    assert state["terminal_classification"] == "GPU_ENGINEERING_BLOCKED_R2"
    with pytest.raises((controller.R2StateError, ValueError)):
        controller.transition(root, "R2_GPU_NUMERICAL_GO")
