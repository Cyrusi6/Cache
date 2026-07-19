from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from script.analysis import phase2a_2a_cache_geometry_pilot as pilot


def test_frozen_fit_members_are_exact() -> None:
    members, counts, digest = pilot._fit_members()
    assert counts == {"ai2-arc": 351, "openbookqa": 158, "mmlu-redux": 1658}
    assert len(members) == 2167
    assert len(digest) == 64
    assert all(task in counts for task, _subject, _qid in members)


def test_read_fit_csv_skips_non_fit_before_outcome_use(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject", "question_id", *pilot.EXACT_COLUMNS],
        )
        writer.writeheader()
        writer.writerow(
            {
                "subject": "allowed",
                "question_id": "1",
                "pred": "A",
                "is_correct": "True",
                "cot_pred": "A",
                "cot_output": "A",
                "cot_gen_length": "1",
            }
        )
        writer.writerow(
            {
                "subject": "sealed",
                "question_id": "2",
                "pred": "SECRET",
                "is_correct": "SECRET",
                "cot_pred": "SECRET",
                "cot_output": "SECRET",
                "cot_gen_length": "SECRET",
            }
        )
    rows = pilot._read_fit_csv(
        path,
        dataset="ai2-arc",
        fit_members={("ai2-arc", "allowed", "1")},
    )
    assert list(rows) == [("ai2-arc", "allowed", "1")]
    assert "SECRET" not in json.dumps(list(rows.values()))


def test_validate_run_rejects_checkpoint_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    config = tmp_path / "eval.yaml"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"checkpoint")
    config.write_text(
        "model:\n  rosetta_config:\n    checkpoints_dir: " + str(checkpoint) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(pilot.CHECKPOINTS, "tinyllama", checkpoint)
    with pytest.raises(pilot.PilotError, match="checkpoint changed"):
        pilot._validate_run(
            {
                "config": str(config),
                "config_sha256": pilot._sha256(config),
                "pair": "tinyllama",
                "checkpoint_sha256": "0" * 64,
            }
        )
