from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import pytest

from script.analysis import route1_identifiability_report as report
from script.analysis import route1_identifiability_suite as suite


generate_report = report.generate_report


def _write_predictions(
    path: Path,
    correctness: Iterable[bool],
    **diagnostics: list[object],
) -> None:
    correctness = list(correctness)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject", "question_id", "is_correct", *diagnostics]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, correct in enumerate(correctness):
            row: dict[str, object] = {
                "subject": "subject-a",
                "question_id": index,
                "is_correct": correct,
            }
            for name, values in diagnostics.items():
                row[name] = values[index]
            writer.writerow(row)


def _write_predictions_with_ids(
    path: Path,
    ids: Iterable[int],
    correctness: Iterable[bool],
    **diagnostics: list[object],
) -> None:
    ids = list(ids)
    correctness = list(correctness)
    assert len(ids) == len(correctness)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject", "question_id", "is_correct", *diagnostics]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, (question_id, correct) in enumerate(zip(ids, correctness)):
            row: dict[str, object] = {
                "subject": "subject-a",
                "question_id": question_id,
                "is_correct": correct,
            }
            for name, values in diagnostics.items():
                row[name] = values[row_index]
            writer.writerow(row)


def _find(rows: list[dict], **expected: object) -> dict:
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in expected.items())
    ]
    assert len(matches) == 1, (expected, matches)
    return matches[0]


def _gate_stats(mean: float) -> dict[str, object]:
    return {
        "count": 10,
        "mean": mean,
        "variance": 0.01,
        "std": 0.1,
        "minimum": 0.1,
        "maximum": 0.9,
        "saturation_low_rate": 0.1,
        "saturation_high_rate": 0.2,
    }


def test_report_computes_transfer_aggregate_paired_and_seed_statistics(
    tmp_path: Path,
) -> None:
    runs = []
    values = {
        (42, "task-a", "B0"): [True, False, True, False],
        (42, "task-a", "B2"): [True, False, False, True],
        (42, "task-a", "B3"): [True, True, False, True],
        (42, "task-b", "B0"): [True, False],
        (42, "task-b", "B2"): [True, True],
        (42, "task-b", "B3"): [True, True],
        (43, "task-a", "B0"): [True, False, True, False],
        (43, "task-a", "B2"): [True, True, False, True],
        (43, "task-a", "B3"): [True, True, True, True],
        (43, "task-b", "B0"): [True, False],
        (43, "task-b", "B2"): [True, False],
        (43, "task-b", "B3"): [True, True],
    }
    for (seed, task, method), correctness in values.items():
        path = tmp_path / "predictions" / f"{method}-{seed}-{task}.csv"
        _write_predictions(path, correctness)
        runs.append(
            {
                "method": method,
                "pair": "TinyLlama->Qwen3",
                "seed": seed,
                "task": task,
                "csv": str(path.relative_to(tmp_path)),
            }
        )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    first_output = tmp_path / "report-first"
    second_output = tmp_path / "report-second"
    first = generate_report(
        manifest,
        first_output,
        bootstrap_samples=500,
        bootstrap_seed=17,
    )
    second = generate_report(
        manifest,
        second_output,
        bootstrap_samples=500,
        bootstrap_seed=17,
    )

    b2_task_a = _find(first["task_metrics"], method="B2", seed=42, task="task-a")
    assert b2_task_a["accuracy"] == 0.5
    assert b2_task_a["positive_transfer_count"] == 1
    assert b2_task_a["positive_transfer_rate"] == 0.5
    assert b2_task_a["negative_transfer_count"] == 1
    assert b2_task_a["negative_transfer_rate"] == 0.5

    b2_aggregate = _find(first["aggregate_metrics"], method="B2", seed=42)
    assert b2_aggregate["macro_mean"] == 0.75
    assert math.isclose(b2_aggregate["weighted_mean"], 4 / 6)

    soft = _find(
        first["paired_comparisons"],
        comparison="soft_candidates",
        seed=42,
        task="task-a",
    )
    assert soft["n_paired"] == 4
    assert soft["delta_accuracy"] == 0.25
    assert soft["improvements"] == 1
    assert soft["regressions"] == 0
    assert soft["mcnemar_exact_p"] == 1.0

    b2_macro = _find(
        first["seed_summary"],
        method="B2",
        task="__aggregate__",
        metric="macro_mean",
    )
    assert b2_macro["n_seeds"] == 2
    assert math.isclose(b2_macro["mean"], 0.6875)
    assert math.isclose(b2_macro["sample_std"], math.sqrt(0.0078125))

    soft_all = _find(
        first["component_contributions"],
        pair="TinyLlama->Qwen3",
        component="soft_candidates",
        seed="all",
    )
    assert soft_all["n_seeds"] == 2
    assert soft_all["positive_seed_count"] == 2
    assert soft_all["bootstrap_ci_low"] is not None
    assert "inconclusive because the corresponding CI crosses or touches zero" in (
        "\n".join(first["mechanism_conclusions"])
    )

    assert first["paired_comparisons"] == second["paired_comparisons"]
    assert first["clustered_comparisons"] == second["clustered_comparisons"]
    assert (first_output / "summary.json").is_file()
    assert (first_output / "task_metrics.csv").is_file()
    assert (first_output / "component_contributions.csv").is_file()
    assert (first_output / "report.md").is_file()
    assert "candidate_count" in {row["field"] for row in first["missing_diagnostics"]}


def test_report_derives_alignment_buckets_and_emits_correlations_and_gate_stats(
    tmp_path: Path,
) -> None:
    receiver = tmp_path / "b0.csv"
    fused = tmp_path / "b6.csv"
    _write_predictions(receiver, [False, False, False, True, True, True])
    _write_predictions(
        fused,
        [False, True, True, True, False, True],
        candidate_count=[1, 2, 4, 1, 3, 1],
        alignment_entropy=[0.0, 0.2, 0.8, 0.1, 0.5, 1.0],
        boundary_mismatch=[0, 1, 3, 0, 2, 4],
        confidence=[0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        gate=[0.0, 0.2, 0.4, 0.8, 0.95, 1.0],
    )
    posthoc = tmp_path / "gate_diagnostics.json"
    posthoc.write_text(
        json.dumps(
            {
                "status": "ok",
                "metadata": {"processed_samples": 64},
                "counts": {
                    "examples_seen": 64,
                    "examples_with_gate": 64,
                    "token_head_gate_projectors": 2,
                },
                "global": {
                    "combined": _gate_stats(0.5),
                    "key": _gate_stats(0.4),
                    "value": _gate_stats(0.6),
                },
                "by_layer": [
                    {
                        "layer": 0,
                        "stage": "early",
                        "key": _gate_stats(0.3),
                        "value": _gate_stats(0.5),
                    },
                    {
                        "layer": 1,
                        "stage": "late",
                        "key": _gate_stats(0.4),
                        "value": _gate_stats(0.7),
                    },
                ],
                "by_stage": [
                    {
                        "stage": "early",
                        "key": _gate_stats(0.3),
                        "value": _gate_stats(0.5),
                    },
                    {
                        "stage": "middle",
                        "key": _gate_stats(0.35),
                        "value": _gate_stats(0.6),
                    },
                    {
                        "stage": "late",
                        "key": _gate_stats(0.4),
                        "value": _gate_stats(0.7),
                    },
                ],
                "by_layer_head": [
                    {
                        "layer": 0,
                        "stage": "early",
                        "head": 0,
                        "key": _gate_stats(0.3),
                        "value": _gate_stats(0.5),
                    }
                ],
                "by_relative_token_bin": [
                    {
                        "relative_token_bin": 0,
                        "key": _gate_stats(0.2),
                        "value": _gate_stats(0.4),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "method": "B0 Receiver-only",
                        "pair": "TinyLlama->Qwen3",
                        "seed": 42,
                        "task": "mmlu-redux",
                        "csv": receiver.name,
                    },
                    {
                        "method": "B6 full",
                        "pair": "TinyLlama->Qwen3",
                        "seed": 42,
                        "task": "mmlu-redux",
                        "csv": fused.name,
                        "gate_diagnostics_posthoc": posthoc.name,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = generate_report(
        manifest,
        tmp_path / "report",
        bootstrap_samples=100,
    )

    one_to_many = _find(
        summary["bucket_metrics"],
        method="B6 full",
        field="alignment_bucket",
        bucket="one-to-many",
    )
    assert one_to_many["status"] == "derived"
    assert one_to_many["n"] == 3
    four_plus = _find(
        summary["bucket_metrics"],
        method="B6 full",
        field="candidate_count",
        bucket="4+",
    )
    assert four_plus["n"] == 1

    confidence_positive = _find(
        summary["correlations"],
        method="B6 full",
        field="confidence",
        outcome="positive_transfer",
    )
    assert confidence_positive["n"] == 3
    assert confidence_positive["status"] == "ok"
    assert confidence_positive["pearson_r"] is not None

    gate = _find(summary["gate_statistics"], method="B6 full")
    assert gate["status"] == "ok"
    assert gate["n"] == 6
    assert math.isclose(gate["mean"], 0.5583333333333333)
    assert math.isclose(gate["saturation_low_rate"], 1 / 6)
    assert math.isclose(gate["saturation_high_rate"], 2 / 6)

    coverage = _find(
        summary["gate_posthoc_coverage"], method="B6 full", seed=42
    )
    assert coverage["status"] == "ok"
    assert coverage["examples_with_gate"] == 64
    early_key = _find(
        summary["gate_posthoc_statistics"],
        method="B6 full",
        scope="stage",
        stage="early",
        kv="key",
    )
    assert early_key["mean"] == 0.3
    assert any(
        row["scope"] == "layer_head"
        for row in summary["gate_posthoc_statistics"]
    )
    assert any(
        row["scope"] == "relative_token_bin"
        for row in summary["gate_posthoc_statistics"]
    )
    assert "K/V stage statistics" in "\n".join(summary["mechanism_conclusions"])
    assert (tmp_path / "report" / "gate_posthoc_statistics.csv").is_file()

    missing_for_b6 = {
        row["field"]
        for row in summary["missing_diagnostics"]
        if row["method"] == "B6 full"
    }
    assert "alignment_bucket" not in missing_for_b6
    assert not missing_for_b6


def test_cli_reports_manifest_contract_errors_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "invalid.json"
    manifest.write_text(json.dumps({"runs": [{"method": "B2"}]}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "route1_identifiability_report.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "report"),
        ],
    )

    with pytest.raises(SystemExit) as error:
        report.main()

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "Each manifest entry needs method, pair, and task" in stderr
    assert "Traceback" not in stderr


def test_receiver_seed42_reuse_clustered_inference_and_final_gate(
    tmp_path: Path,
) -> None:
    runs: list[dict[str, object]] = []
    receiver = tmp_path / "predictions" / "receiver-seed42.csv"
    _write_predictions(receiver, [False] * 10 + [True] * 10)
    runs.append(
        {
            "method": "B0",
            "pair": "receiver",
            "seed": 42,
            "task": "mmlu-redux",
            "csv": str(receiver.relative_to(tmp_path)),
        }
    )
    runs.append(
        {
            "method": "B0",
            "pair": "pair-1",
            "seed": 42,
            "task": "mmlu-redux",
            "csv": str(receiver.relative_to(tmp_path)),
        }
    )

    for pair_index in range(1, 5):
        pair = f"pair-{pair_index}"
        for seed in (42, 43, 44):
            if pair_index <= 3:
                values = {
                    "B2": [False] * 20,
                    "B2-constant": [False] * 20,
                    "B5": [True] * 10 + [False] * 10,
                    "B6": [True] * 20,
                }
            else:
                values = {
                    method: [True] * 20 for method in ("B2", "B2-constant", "B5", "B6")
                }
            for task in ("ai2-arc", "openbookqa", "mmlu-redux"):
                for method, correctness in values.items():
                    path = (
                        tmp_path
                        / "predictions"
                        / f"{pair}-{method}-{seed}-{task}.csv"
                    )
                    _write_predictions(path, correctness)
                    runs.append(
                        {
                            "method": method,
                            "pair": pair,
                            "seed": seed,
                            "task": task,
                            "csv": str(path.relative_to(tmp_path)),
                        }
                    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "report_contract": {
                    "required_pairs": [f"pair-{index}" for index in range(1, 5)],
                    "required_seeds": [42, 43, 44],
                    "expected_task_rows": {
                        "ai2-arc": 20,
                        "openbookqa": 20,
                        "mmlu-redux": 20,
                    },
                },
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    summary = generate_report(
        manifest,
        tmp_path / "report",
        bootstrap_samples=1000,
        bootstrap_seed=91,
    )

    reused = _find(
        summary["task_metrics"],
        pair="pair-1",
        method="B6",
        seed=44,
        task="mmlu-redux",
    )
    assert reused["transfer_status"] == "ok"
    assert reused["receiver_source_kind"] == "manifest_run"
    assert reused["receiver_source_pair"] == "pair-1"
    assert reused["receiver_source_seed"] == 42
    assert reused["receiver_seed_reused"] is True

    across_seeds = _find(
        summary["clustered_comparisons"],
        comparison="full_over_hard_span",
        aggregation_level="across_seeds_within_pair",
        cluster_unit="pair_seed",
        pair="pair-1",
    )
    assert across_seeds["n_clusters"] == 3
    assert across_seeds["n_seeds"] == 3
    assert across_seeds["bootstrap_ci_low"] > 0

    across_pairs = _find(
        summary["clustered_comparisons"],
        comparison="full_over_hard_span",
        aggregation_level="across_pairs",
        cluster_unit="pair",
        pair="__all__",
    )
    assert across_pairs["n_pairs"] == 4
    assert across_pairs["positive_pair_count"] == 3
    assert across_pairs["bootstrap_ci_low"] > 0
    assert across_pairs["aggregate_mcnemar_exact_p"] < 0.001
    assert across_pairs["aggregate_mcnemar_scope"].endswith("not cluster-adjusted")

    b6_b2_gate = _find(summary["final_gate"], contrast="B6_vs_B2")
    b6_b5_gate = _find(summary["final_gate"], contrast="B6_vs_B5")
    combined_gate = _find(summary["final_gate"], contrast="combined_B6_vs_B2_and_B5")
    assert b6_b2_gate["status"] == "pass"
    assert b6_b5_gate["status"] == "pass"
    assert combined_gate["gate_pass"] is True

    clean_gate = _find(
        summary["component_contributions"],
        pair="pair-1",
        component="gate_capacity",
        seed="all",
    )
    assert clean_gate["baseline_method"] == "B2-constant"
    assert clean_gate["bootstrap_ci_low"] > 0
    assert not any(
        row["component"] == "gate_capacity_confounded"
        for row in summary["component_contributions"]
    )


@pytest.mark.parametrize(
    ("candidate_ids", "expected_rows", "expected_status"),
    [
        ([0, 2], 2, "sample_keys_mismatch"),
        ([0, 1], 3, "unexpected_n"),
    ],
)
def test_incomplete_pairing_is_reported_but_excluded_from_aggregation(
    tmp_path: Path,
    candidate_ids: list[int],
    expected_rows: int,
    expected_status: str,
) -> None:
    baseline = tmp_path / "b2.csv"
    candidate = tmp_path / "b6.csv"
    _write_predictions_with_ids(baseline, [0, 1], [False, False])
    _write_predictions_with_ids(candidate, candidate_ids, [True, True])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "report_contract": {
                    "required_pairs": ["pair-1", "pair-2", "pair-3", "pair-4"],
                    "required_seeds": [42, 43, 44],
                    "expected_task_rows": {"task-a": expected_rows},
                },
                "runs": [
                    {
                        "method": "B2",
                        "pair": "pair-1",
                        "seed": 42,
                        "task": "task-a",
                        "csv": baseline.name,
                    },
                    {
                        "method": "B6",
                        "pair": "pair-1",
                        "seed": 42,
                        "task": "task-a",
                        "csv": candidate.name,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = generate_report(manifest, tmp_path / "report", bootstrap_samples=50)
    row = _find(
        summary["paired_comparisons"],
        comparison="full_over_hard_span",
        task="task-a",
    )
    assert row["pairing_status"] == expected_status
    assert row["aggregation_eligible"] is False
    assert row["delta_accuracy"] is None
    assert not any(
        item["comparison"] == "full_over_hard_span"
        for item in summary["clustered_comparisons"]
    )
    assert _find(summary["final_gate"], contrast="B6_vs_B2")["status"] == (
        "incomplete"
    )


def test_missing_required_task_and_seed_cannot_pass_final_gate(tmp_path: Path) -> None:
    runs: list[dict[str, object]] = []
    for pair_index in range(1, 5):
        pair = f"pair-{pair_index}"
        for method, correctness in {"B2": [False, False], "B6": [True, True]}.items():
            path = tmp_path / f"{pair}-{method}.csv"
            _write_predictions(path, correctness)
            runs.append(
                {
                    "method": method,
                    "pair": pair,
                    "seed": 42,
                    "task": "task-a",
                    "csv": path.name,
                }
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "report_contract": {
                    "required_pairs": [f"pair-{index}" for index in range(1, 5)],
                    "required_seeds": [42, 43, 44],
                    "expected_task_rows": {"task-a": 2, "task-b": 2},
                },
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    summary = generate_report(manifest, tmp_path / "report", bootstrap_samples=50)
    gate = _find(summary["final_gate"], contrast="B6_vs_B2")
    assert gate["available_pair_count"] == 0
    assert gate["coverage_complete"] is False
    assert gate["status"] == "incomplete"


def test_across_pair_bootstrap_equal_weights_pairs_then_seeds(tmp_path: Path) -> None:
    runs: list[dict[str, object]] = []
    for pair_index in range(1, 5):
        pair = f"pair-{pair_index}"
        for seed, size, baseline_value, candidate_value in (
            (42, 100, False, True),
            (43, 1, True, False),
            (44, 1, True, False),
        ):
            for method, value in (("B2", baseline_value), ("B6", candidate_value)):
                path = tmp_path / f"{pair}-{seed}-{method}.csv"
                _write_predictions(path, [value] * size)
                runs.append(
                    {
                        "method": method,
                        "pair": pair,
                        "seed": seed,
                        "task": "task-a",
                        "csv": path.name,
                    }
                )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    summary = generate_report(
        manifest, tmp_path / "report", bootstrap_samples=500, bootstrap_seed=19
    )
    row = _find(
        summary["clustered_comparisons"],
        comparison="full_over_hard_span",
        aggregation_level="across_pairs",
        cluster_unit="pair",
    )
    assert math.isclose(row["delta_accuracy"], -1 / 3)
    assert row["pooled_accuracy_delta"] > 0.9
    assert row["bootstrap_level"] == "pairs_then_seeds_then_paired_examples"
    assert row["bootstrap_ci_low"] < 0


def test_paired_bucket_gains_use_candidate_diagnostics_on_fixed_keys(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "b2.csv"
    candidate = tmp_path / "b3.csv"
    _write_predictions(
        baseline,
        [False, False, False, False],
        candidate_count=[1, 1, 1, 1],
    )
    _write_predictions(
        candidate,
        [False, False, True, True],
        candidate_count=[1, 1, 2, 2],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "method": "B2",
                        "pair": "tinyllama",
                        "seed": 42,
                        "task": "task-a",
                        "csv": baseline.name,
                    },
                    {
                        "method": "B3",
                        "pair": "tinyllama",
                        "seed": 42,
                        "task": "task-a",
                        "csv": candidate.name,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    summary = generate_report(manifest, tmp_path / "report", bootstrap_samples=100)
    one = _find(
        summary["paired_bucket_gains"],
        comparison="soft_candidates",
        task="task-a",
        field="candidate_count",
        bucket="1",
    )
    two = _find(
        summary["paired_bucket_gains"],
        comparison="soft_candidates",
        task="task-a",
        field="candidate_count",
        bucket="2",
    )
    assert one["bucket_source_method"] == "B3"
    assert one["delta_accuracy"] == 0.0
    assert two["n_paired"] == 2
    assert two["delta_accuracy"] == 1.0
    assert (tmp_path / "report" / "paired_bucket_gains.csv").is_file()


def test_suite_materialized_manifest_is_supported_end_to_end(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_values = {
        ("receiver", "b0", 42): [True, False, True, False],
        ("tinyllama", "b2", 42): [True, False, False, True],
        ("tinyllama", "b5", 42): [True, True, False, True],
        ("tinyllama", "b6", 42): [True, True, True, True],
        ("tinyllama", "b2", 43): [True, False, False, True],
        ("tinyllama", "b5", 43): [True, True, False, True],
        ("tinyllama", "b6", 43): [True, True, True, True],
    }
    suite_runs = []
    for (pair, variant, seed), correctness in run_values.items():
        output_dir = artifacts / pair / variant / f"seed_{seed}" / "mmlu-redux"
        prediction = output_dir / f"{variant}_cot.csv"
        _write_predictions(prediction, correctness)
        suite_runs.append(
            {
                "run_id": f"{pair}__{variant}__seed_{seed}",
                "pair": pair,
                "variant": variant,
                "seed": seed,
                "datasets": {
                    "mmlu-redux": {
                        "output_dir": str(output_dir),
                        "prediction_glob": str(output_dir / "*_cot.csv"),
                    }
                },
            }
        )

    manifest = tmp_path / "analysis_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receiver_baseline_run_id": "receiver__b0__seed_42",
                "component_comparisons": [
                    {
                        "question": "gate_capacity",
                        "candidate": "b5",
                        "control": "b2",
                    },
                    {
                        "question": "gate_soft_interaction",
                        "candidate": "b6",
                        "control": "b5",
                    },
                ],
                "runs": suite_runs,
            }
        ),
        encoding="utf-8",
    )

    direct_summary = generate_report(
        manifest,
        tmp_path / "suite-direct-report",
        bootstrap_samples=200,
    )
    assert (
        _find(
            direct_summary["task_metrics"],
            method="b6",
            pair="tinyllama",
            seed=43,
            task="mmlu-redux",
        )["receiver_source_seed"]
        == 42
    )

    materialized = tmp_path / "report_input_manifest.json"
    suite.materialize_analysis_manifest(
        analysis_manifest_path=manifest,
        output_path=materialized,
        allow_missing=True,
    )
    summary = generate_report(
        materialized,
        tmp_path / "suite-materialized-report",
        bootstrap_samples=200,
    )

    seed43 = _find(
        summary["task_metrics"],
        method="B6",
        pair="tinyllama",
        seed=43,
        task="mmlu-redux",
    )
    assert seed43["receiver_source_kind"] == "receiver_csv"
    assert seed43["transfer_status"] == "ok"
    hard_span = _find(
        summary["paired_comparisons"],
        comparison="full_over_hard_span",
        seed=43,
        task="mmlu-redux",
    )
    assert hard_span["candidate_code"] == "B6"
    assert hard_span["baseline_code"] == "B2"
    confounded = _find(
        summary["component_contributions"],
        pair="tinyllama",
        component="gate_capacity_confounded",
        seed="all",
    )
    assert confounded["baseline_method"] == "B2"
    assert "confounded by confidence mismatch" in "\n".join(
        summary["mechanism_conclusions"]
    )
    assert _find(summary["final_gate"], contrast="B6_vs_B2")["status"] == ("incomplete")
