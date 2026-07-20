import json
from pathlib import Path

import pytest

from script.experiment.fpct_confirmatory_controller import StateError, initialize, record_triplet, transition


def make_lock(tmp_path: Path) -> Path:
    path = tmp_path / "lock.json"
    path.write_text(json.dumps({"run_uid": "fpct-test"}) + "\n")
    return path


def test_controller_is_idempotent_and_requires_complete_triplets(tmp_path):
    root = tmp_path / "run"
    run_lock = make_lock(tmp_path)
    state = initialize(root, run_lock)
    assert initialize(root, run_lock) == state
    for stage in ("GPU_NUMERICAL_GO", "PRETRAINED_SMOKE_GO", "MATCHED_SMOKE_GO", "FORMAL_TRAINING_SUBMITTED"):
        transition(root, stage)
    with pytest.raises(StateError, match="12 complete"):
        transition(root, "FORMAL_TRAINING_COMPLETE")


def test_all_triplets_release_held_out_once(tmp_path):
    root = tmp_path / "run"; initialize(root, make_lock(tmp_path))
    for stage in ("GPU_NUMERICAL_GO", "PRETRAINED_SMOKE_GO", "MATCHED_SMOKE_GO", "FORMAL_TRAINING_SUBMITTED"):
        transition(root, stage)
    for seed in range(45, 57):
        evidence = tmp_path / f"{seed}.json"; evidence.write_text("{}\n")
        record_triplet(root, seed, evidence, "COMPLETE")
    transition(root, "FORMAL_TRAINING_COMPLETE")
    transition(root, "MODEL_SELECTION_RELEASED")
    transition(root, "MODEL_SELECTION_COMPLETE")
    state = transition(root, "HELD_OUT_RELEASED")
    assert state["held_out_release_count"] == 1
