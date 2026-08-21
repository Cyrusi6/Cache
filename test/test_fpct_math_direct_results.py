from __future__ import annotations

import csv
import json
from pathlib import Path

from script.analysis.fpct_math_direct_results import SEED, _content_hash, reduce_results


def test_math_direct_four_cell_reduction_and_integrity(tmp_path: Path) -> None:
    counts = {"ai2-arc": 128, "openbookqa": 70, "mmlu-redux": 128}
    rows = []
    for task, count in counts.items():
        for index in range(count):
            question = f"{task} question {index}"
            choices = [f"choice {letter} {index}" for letter in "ABCD"]
            rows.append(
                {
                    "task": task,
                    "evaluation_subject": "subject",
                    "evaluation_question_id": index,
                    "content_group_sha256": _content_hash(question, choices),
                    "question": question,
                    "choices": choices,
                }
            )
    dev_path = tmp_path / "dev.json"
    dev_path.write_text(json.dumps({"group_counts": counts, "rows": rows}))

    shared = {
        "step0_trainable_sha256": "step0",
        "trainable_keys_sha256": "keys",
        "data_order_sha256": "order",
        "rng_state_before_training_sha256": "rng",
        "optimizer_class": "torch.optim.AdamW",
        "optimizer_group_count": 3,
        "scheduler_initial_state_sha256": "scheduler",
        "optimizer_steps": 64,
        "checkpoint_reload_equal": True,
    }
    for arm in ("c_post", "f"):
        path = tmp_path / "seeds" / str(SEED) / arm
        path.mkdir(parents=True)
        record = {
            **shared,
            "operator": arm,
            "official_checkpoint_sha256": f"checkpoint-{arm}",
        }
        (path / "fpct_formal_integrity.json").write_text(json.dumps(record))

    correctness = {"Y_CC": False, "Y_CF": True, "Y_FC": False, "Y_FF": True}
    for cell, is_correct in correctness.items():
        for task in counts:
            path = tmp_path / "seeds" / str(SEED) / "eval" / cell / task
            path.mkdir(parents=True)
            task_rows = [row for row in rows if row["task"] == task]
            with (path / "results_cot.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "subject",
                        "question_id",
                        "question",
                        "A",
                        "B",
                        "C",
                        "D",
                        "is_correct",
                    ],
                )
                writer.writeheader()
                for row in task_rows:
                    writer.writerow(
                        {
                            "subject": "subject",
                            "question_id": row["evaluation_question_id"],
                            "question": row["question"],
                            **dict(zip("ABCD", row["choices"])),
                            "is_correct": str(is_correct),
                        }
                    )

    result = reduce_results(tmp_path, dev_path)
    assert result["classification"] == "SINGLE_SEED_POSITIVE"
    assert result["estimands"] == {
        "T": 1.0,
        "D_C": 1.0,
        "D_F": 1.0,
        "O": 1.0,
        "I": 0.0,
    }
    assert result["matched_integrity"]["status"] == "GO"
