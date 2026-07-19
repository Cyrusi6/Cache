from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from script.analysis.phase2a_0_opportunity_audit import run_audit


FIELDS = [
    "subject",
    "question_id",
    "question",
    "A",
    "B",
    "C",
    "D",
    "true_answer",
    "pred",
    "is_correct",
    "cot_input_length",
    "alignment_bucket",
    "candidate_count",
    "candidate_count_max",
    "one_to_many_rate",
    "alignment_entropy",
    "boundary_mismatch",
    "confidence",
    "fallback_rate",
]


def _write_predictions(path: Path, values: list[tuple[str, str, bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for subject, question_id, correct in values:
            writer.writerow(
                {
                    "subject": subject,
                    "question_id": question_id,
                    "question": f"Question {subject}/{question_id}",
                    "A": "a",
                    "B": "b",
                    "C": "c",
                    "D": "d",
                    "true_answer": "A",
                    "pred": "A" if correct else "B",
                    "is_correct": str(correct),
                    "cot_input_length": "10",
                    "alignment_bucket": "one-to-many",
                    "candidate_count": "2",
                    "candidate_count_max": "2",
                    "one_to_many_rate": "0.5",
                    "alignment_entropy": "0.5",
                    "boundary_mismatch": "0",
                    "confidence": "0.75",
                    "fallback_rate": "0",
                }
            )


def _build_fixture(tmp_path: Path) -> Path:
    receiver_values = [("s1", "0", True), ("s1", "1", False), ("s2", "0", True), ("s2", "1", False)]
    fused = {
        ("p1", 1): [("s1", "0", True), ("s1", "1", True), ("s2", "0", False), ("s2", "1", False)],
        ("p1", 2): [("s1", "0", True), ("s1", "1", True), ("s2", "0", False), ("s2", "1", True)],
        ("p2", 1): [("s1", "0", False), ("s1", "1", True), ("s2", "0", True), ("s2", "1", False)],
        ("p2", 2): [("s1", "0", True), ("s1", "1", False), ("s2", "0", True), ("s2", "1", False)],
    }
    receiver_path = tmp_path / "results/receiver/b0/seed_1/task/receiver_cot.csv"
    _write_predictions(receiver_path, receiver_values)
    runs = [
        {
            "run_id": "receiver__b0__seed_1",
            "pair": "receiver",
            "variant": "b0",
            "seed": 1,
            "datasets": {
                "task": {"prediction_glob": "results/receiver/b0/seed_1/task/*_cot.csv"}
            },
        }
    ]
    for (pair, seed), values in fused.items():
        path = tmp_path / f"results/{pair}/b6/seed_{seed}/task/pred_cot.csv"
        _write_predictions(path, values)
        runs.append(
            {
                "run_id": f"{pair}__b6__seed_{seed}",
                "pair": pair,
                "variant": "b6",
                "seed": seed,
                "datasets": {
                    "task": {
                        "prediction_glob": f"results/{pair}/b6/seed_{seed}/task/*_cot.csv"
                    }
                },
            }
        )
    analysis = {"receiver_baseline_run_id": "receiver__b0__seed_1", "runs": runs}
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    import hashlib

    manifest = {
        "schema_version": 1,
        "source_commit": "test",
        "constraints": {
            "gpu": False,
            "training": False,
            "checkpoint_mutation": False,
            "selector_training": False,
        },
        "source": {
            "artifact_root": str(tmp_path),
            "artifact_commit": "test",
            "phase1_analysis_manifest": "analysis.json",
            "phase1_analysis_manifest_sha256": hashlib.sha256(
                analysis_path.read_bytes()
            ).hexdigest(),
            "receiver_run_id": "receiver__b0__seed_1",
            "fused_variant": "b6",
        },
        "pairs": [
            {"id": "p1", "heterogeneous": True},
            {"id": "p2", "heterogeneous": False},
        ],
        "seeds": [1, 2],
        "tasks": [{"id": "task", "expected_rows": 4}],
        "bootstrap": {"samples": 200, "confidence": 0.95, "seed": 7, "batch_size": 50},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_audit_builds_four_events_and_true_best_fixed(tmp_path: Path) -> None:
    manifest = _build_fixture(tmp_path)
    result = run_audit(
        manifest,
        tmp_path / "aggregate.csv",
        tmp_path / "aggregate.json",
    )
    row = next(
        item
        for item in result["aggregate_rows"]
        if item["aggregation_level"] == "pair_balanced"
        and item["pair"] == "__all__"
        and item["weighting"] == "sample_weighted"
    )
    assert row["receiver_correct_fused_correct_rate"] == pytest.approx(0.3125)
    assert row["beneficial_transfer_rate"] == pytest.approx(0.25)
    assert row["harmful_transfer_rate"] == pytest.approx(0.1875)
    assert row["receiver_wrong_fused_wrong_rate"] == pytest.approx(0.25)
    assert row["receiver_accuracy"] == pytest.approx(0.5)
    assert row["fused_accuracy"] == pytest.approx(0.5625)
    assert row["oracle_accuracy"] == pytest.approx(0.75)
    assert row["oracle_headroom_over_best_fixed"] == pytest.approx(0.1875)
    assert row["best_fixed_policy"] == "fused"
    assert sum(row[name] for name in EVENT_NAMES_FOR_TEST) == pytest.approx(1.0)


EVENT_NAMES_FOR_TEST = (
    "receiver_correct_fused_correct_rate",
    "beneficial_transfer_rate",
    "harmful_transfer_rate",
    "receiver_wrong_fused_wrong_rate",
)


def test_audit_is_deterministic(tmp_path: Path) -> None:
    manifest = _build_fixture(tmp_path)
    first = run_audit(manifest, tmp_path / "a.csv", tmp_path / "a.json")
    second = run_audit(manifest, tmp_path / "b.csv", tmp_path / "b.json")
    assert first["aggregate_rows"] == second["aggregate_rows"]
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_audit_rejects_content_mismatch(tmp_path: Path) -> None:
    manifest = _build_fixture(tmp_path)
    path = tmp_path / "results/p1/b6/seed_1/task/pred_cot.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    rows[0]["question"] = "different input"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Input-content mismatch"):
        run_audit(manifest, tmp_path / "bad.csv", tmp_path / "bad.json")
