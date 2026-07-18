from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from script.analysis.phase1_5_causal_diagnostics import (
    _load_manifest,
    generate_diagnostics,
)


def _write_predictions(
    path: Path,
    correctness: Iterable[bool],
    diagnostics: Mapping[str, Iterable[object]] | None = None,
) -> None:
    values = list(correctness)
    diagnostic_values = {
        name: list(items) for name, items in (diagnostics or {}).items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["subject", "question_id", "is_correct", *diagnostic_values]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, correct in enumerate(values):
            row: dict[str, object] = {
                "subject": "subject-a",
                "question_id": index,
                "is_correct": correct,
            }
            for name, items in diagnostic_values.items():
                row[name] = items[index]
            writer.writerow(row)


def _write_token_diagnostics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject",
                "question_id",
                "token_index",
                "candidate_count",
                "alignment_entropy",
                "boundary_mismatch",
            ],
        )
        writer.writeheader()
        token_rows = {
            0: [(1, 0.0, 0.0), (1, 0.0, 0.0)],
            1: [(1, 0.0, 0.0), (1, 0.0, 0.0)],
            2: [(1, 0.0, 0.0), (2, 2.0, 1.0)],
            3: [(2, 3.0, 0.0), (2, 3.0, 0.0)],
        }
        for question_id, rows in token_rows.items():
            for token_index, (count, entropy, mismatch) in enumerate(rows):
                writer.writerow(
                    {
                        "subject": "subject-a",
                        "question_id": question_id,
                        "token_index": token_index,
                        "candidate_count": count,
                        "alignment_entropy": entropy,
                        "boundary_mismatch": mismatch,
                    }
                )


def _find(rows: list[dict], **expected: object) -> dict:
    matches = [
        row
        for row in rows
        if all(row.get(name) == value for name, value in expected.items())
    ]
    assert len(matches) == 1, (expected, matches)
    return matches[0]


def test_oracle_headroom_and_fixed_native_ambiguity_source(tmp_path: Path) -> None:
    receiver = tmp_path / "receiver.csv"
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    native = tmp_path / "native.csv"
    token_diagnostics = tmp_path / "native-token-diagnostics.csv"
    _write_predictions(receiver, [True, True, False, False])
    _write_predictions(baseline, [True, False, False, False])
    _write_predictions(
        candidate,
        [True, False, True, False],
        {
            # Deliberately conflicts with the native source and must be ignored.
            "candidate_count_max": [4, 1, 1, 1],
            "alignment_entropy": [4.0, 0.0, 0.0, 0.0],
        },
    )
    _write_predictions(native, [False, False, False, False])
    _write_token_diagnostics(token_diagnostics)

    runs = []
    for method, path in (
        ("receiver_only", receiver),
        ("b2_eval_k1", baseline),
        ("b2_eval_k4", candidate),
        ("b3_native", native),
    ):
        run = {
            "method": method,
            "pair": "tinyllama",
            "seed": 42,
            "task": "task-a",
            "csv": path.name,
        }
        if method == "b3_native":
            run["ambiguity_csv"] = token_diagnostics.name
        runs.append(run)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "receiver_method": "receiver_only",
                "oracle_methods": ["b2_eval_k4"],
                "report_contract": {"expected_task_rows": {"task-a": 4}},
                "comparisons": [
                    {
                        "name": "eval_k4_vs_k1",
                        "baseline": "b2_eval_k1",
                        "candidate": "b2_eval_k4",
                        "ambiguity_source": "b3_native",
                    }
                ],
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    summary = generate_diagnostics(
        manifest, tmp_path / "report", bootstrap_samples=300, bootstrap_seed=11
    )

    oracle = _find(
        summary["oracle_abstention"],
        method="b2_eval_k4",
        pair="tinyllama",
        seed=42,
        task="task-a",
    )
    assert oracle["receiver_accuracy"] == 0.5
    assert oracle["fused_accuracy"] == 0.5
    assert oracle["oracle_accuracy"] == 0.75
    assert oracle["oracle_headroom_over_fused"] == 0.25
    assert oracle["oracle_headroom_over_best_fixed"] == 0.25
    assert oracle["ideal_abstain_count"] == 1
    assert oracle["beneficial_transfer_count"] == 1

    absolute = _find(
        summary["ambiguity_interactions"],
        comparison="eval_k4_vs_k1",
        pair="tinyllama",
        seed=42,
        task="task-a",
        scheme="absolute",
    )
    assert absolute["ambiguity_source_method"] == "b3_native"
    assert absolute["ambiguity_source_csv"] == str(token_diagnostics.resolve())
    assert absolute["high_n"] == 2
    assert absolute["low_n"] == 2
    assert absolute["high_delta_accuracy"] == 0.5
    assert absolute["low_delta_accuracy"] == 0.0
    assert absolute["ambiguity_interaction"] == 0.5

    q75 = _find(
        summary["ambiguity_interactions"],
        comparison="eval_k4_vs_k1",
        pair="tinyllama",
        seed=42,
        task="task-a",
        scheme="q75",
    )
    assert q75["high_n"] == 1
    assert q75["low_n"] == 3
    assert q75["high_delta_accuracy"] == 0.0
    assert math.isclose(q75["low_delta_accuracy"], 1 / 3)
    assert math.isclose(q75["ambiguity_interaction"], -1 / 3)


def test_hierarchical_intervention_bootstrap_equal_weights_pairs(
    tmp_path: Path,
) -> None:
    runs: list[dict[str, object]] = []
    for pair, size, baseline_value, candidate_value in (
        ("large-positive", 100, False, True),
        ("small-negative", 1, True, False),
    ):
        for method, value in (
            ("eval_k1", baseline_value),
            ("eval_k4", candidate_value),
        ):
            path = tmp_path / f"{pair}-{method}.csv"
            _write_predictions(path, [value] * size)
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
                "receiver_method": "unused",
                "oracle_methods": [],
                "comparisons": [
                    {
                        "name": "eval_k4_vs_k1",
                        "baseline": "eval_k1",
                        "candidate": "eval_k4",
                    }
                ],
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    summary = generate_diagnostics(
        manifest, tmp_path / "report", bootstrap_samples=400, bootstrap_seed=23
    )
    row = _find(
        summary["hierarchical_interventions"],
        comparison="eval_k4_vs_k1",
        aggregation_level="across_pairs",
        pair="__all__",
    )
    assert row["delta_accuracy"] == 0.0
    assert row["positive_pair_count"] == 1
    assert row["bootstrap_level"] == "pairs_then_seeds_then_paired_examples"
    assert row["bootstrap_ci_low"] < 0.0
    assert row["bootstrap_ci_high"] > 0.0

    variance = _find(
        summary["seed_variance"], comparison="eval_k4_vs_k1", pair="__all__"
    )
    assert variance["n_seeds"] == 1
    assert variance["seed_sample_std"] is None


def test_cli_outputs_reproducible_csv_and_json_artifacts(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_predictions(baseline, [False, True])
    _write_predictions(candidate, [True, True])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "oracle_methods": [],
                "comparisons": [
                    {"name": "candidate_vs_baseline", "baseline": "a", "candidate": "b"}
                ],
                "runs": [
                    {
                        "method": "a",
                        "pair": "p",
                        "seed": 42,
                        "task": "t",
                        "csv": baseline.name,
                    },
                    {
                        "method": "b",
                        "pair": "p",
                        "seed": 42,
                        "task": "t",
                        "csv": candidate.name,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    first = generate_diagnostics(
        manifest, output, bootstrap_samples=50, bootstrap_seed=7
    )
    second = generate_diagnostics(
        manifest, tmp_path / "output-2", bootstrap_samples=50, bootstrap_seed=7
    )
    assert first == second | {"manifest": str(manifest.resolve())}
    assert (output / "paired_interventions.csv").is_file()
    assert (output / "hierarchical_interventions.csv").is_file()
    assert (output / "seed_variance.csv").is_file()
    assert (output / "oracle_abstention.csv").is_file()
    assert (output / "ambiguity_interactions.csv").is_file()
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == first


def test_consumes_phase15_execution_manifest_directly(tmp_path: Path) -> None:
    receiver = tmp_path / "receiver.csv"
    b2_native = tmp_path / "b2-native.csv"
    b3_native = tmp_path / "b3-native.csv"
    b2_k4 = tmp_path / "results" / "b2-k4" / "run_cot.csv"
    b3_k1 = tmp_path / "results" / "b3-k1" / "run_cot.csv"
    _write_predictions(receiver, [True, False, False, True])
    _write_predictions(b2_native, [True, False, False, False])
    _write_predictions(
        b3_native,
        [True, False, True, False],
        {
            "candidate_count_max": [1, 1, 2, 2],
            "alignment_entropy": [0.0, 0.0, 1.0, 2.0],
            "one_to_many_rate": [0.0, 0.0, 1.0, 1.0],
            "boundary_mismatch": [0.0, 0.0, 0.0, 1.0],
        },
    )
    _write_predictions(b2_k4, [True, False, True, False])
    _write_predictions(b3_k1, [True, False, False, False])

    phase1_analysis = tmp_path / "phase1-analysis.json"
    phase1_analysis.write_text(
        json.dumps(
            {
                "report_contract": {"expected_task_rows": {"task-a": 4}},
                "runs": [
                    {
                        "run_id": "receiver",
                        "pair": "receiver",
                        "variant": "b0",
                        "seed": 42,
                        "datasets": {"task-a": {"prediction_glob": receiver.name}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def execution_run(
        run_id: str,
        trained_variant: str,
        intervention_id: str,
        output_dir: Path,
        native_variant: str,
        native_csv: Path,
    ) -> dict[str, object]:
        return {
            "id": run_id,
            "pair": "tinyllama",
            "seed": 42,
            "trained_variant": trained_variant,
            "intervention": {"id": intervention_id},
            "output_dirs": {"task-a": str(output_dir)},
            "native_comparator": {
                "variant": native_variant,
                "prediction_csv": {"task-a": str(native_csv)},
            },
            "ambiguity_source": {
                "variant": "b3",
                "fixed_across_contrast": True,
                "prediction_csv": {"task-a": str(b3_native)},
            },
        }

    execution_manifest = tmp_path / "execution-manifest.json"
    execution_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite": "route1_phase1_5_same_checkpoint_interventions",
                "source": {
                    "phase1_analysis_manifest": str(phase1_analysis),
                    "phase1_artifact_root": str(tmp_path),
                },
                "runs": [
                    execution_run(
                        "b2", "b2", "b2_eval_k4", b2_k4.parent, "b2", b2_native
                    ),
                    execution_run(
                        "b3", "b3", "b3_eval_k1", b3_k1.parent, "b3", b3_native
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = generate_diagnostics(
        execution_manifest,
        tmp_path / "analysis",
        bootstrap_samples=100,
        bootstrap_seed=29,
    )
    b2_effect = _find(
        summary["paired_interventions"],
        comparison="b2_train_k1_eval_k4_vs_k1",
        pair="tinyllama",
        seed=42,
        task="task-a",
    )
    assert b2_effect["delta_accuracy"] == 0.25
    b2_hierarchical = _find(
        summary["hierarchical_interventions"],
        comparison="b2_train_k1_eval_k4_vs_k1",
        aggregation_level="across_pairs",
        pair="__all__",
    )
    assert b2_hierarchical["heterogeneous_pair_count"] == 1
    assert b2_hierarchical["positive_heterogeneous_pair_count"] == 1
    train_at_k1 = _find(
        summary["paired_interventions"],
        comparison="train_k4_vs_k1_at_eval_k1",
        pair="tinyllama",
        seed=42,
        task="task-a",
    )
    assert train_at_k1["delta_accuracy"] == 0.0
    ambiguity = _find(
        summary["ambiguity_interactions"],
        comparison="b2_train_k1_eval_k4_vs_k1",
        pair="tinyllama",
        seed=42,
        task="task-a",
        scheme="absolute",
    )
    assert ambiguity["ambiguity_source_method"] == "b3_native"
    assert ambiguity["high_delta_accuracy"] == 0.5


def test_optional_qwen25_seed44_anomaly_manifest_adds_stats_and_oracle(
    tmp_path: Path,
) -> None:
    receiver = tmp_path / "receiver.csv"
    b2_native = tmp_path / "b2-native.csv"
    b2_k4 = tmp_path / "main-results" / "b2-k4" / "run_cot.csv"
    b6_native = tmp_path / "b6-native.csv"
    alignment_forced = (
        tmp_path / "anomaly-results" / "alignment" / "run_cot.csv"
    )
    legacy_forced = tmp_path / "anomaly-results" / "legacy" / "run_cot.csv"
    _write_predictions(receiver, [True, False, False, True])
    _write_predictions(b2_native, [True, False, False, False])
    _write_predictions(b2_k4, [True, False, True, False])
    _write_predictions(
        b6_native,
        [True, False, True, False],
        {
            "candidate_count_max": [1, 1, 2, 2],
            "alignment_entropy": [0.0, 0.0, 1.0, 2.0],
            "one_to_many_rate": [0.0, 0.0, 1.0, 1.0],
            "boundary_mismatch": [0.0, 0.0, 0.0, 1.0],
        },
    )
    _write_predictions(alignment_forced, [True, True, False, False])
    _write_predictions(legacy_forced, [True, True, True, False])

    phase1_analysis = tmp_path / "phase1-analysis.json"
    phase1_analysis.write_text(
        json.dumps(
            {
                "report_contract": {"expected_task_rows": {"task-a": 4}},
                "runs": [
                    {
                        "run_id": "receiver",
                        "pair": "receiver",
                        "variant": "b0",
                        "seed": 42,
                        "datasets": {"task-a": {"prediction_glob": receiver.name}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    source = {
        "phase1_analysis_manifest": str(phase1_analysis),
        "phase1_artifact_root": str(tmp_path),
    }

    def run(
        *,
        pair: str,
        seed: int,
        trained_variant: str,
        intervention: str,
        output: Path,
        native_variant: str,
        native_csv: Path,
    ) -> dict[str, object]:
        return {
            "id": intervention,
            "pair": pair,
            "seed": seed,
            "trained_variant": trained_variant,
            "intervention": {"id": intervention},
            "output_dirs": {"task-a": str(output.parent)},
            "native_comparator": {
                "variant": native_variant,
                "prediction_csv": {"task-a": str(native_csv)},
            },
            "ambiguity_source": {
                "variant": native_variant,
                "fixed_across_contrast": True,
                "prediction_csv": {"task-a": str(native_csv)},
            },
        }

    main_manifest = tmp_path / "main-manifest.json"
    main_manifest.write_text(
        json.dumps(
            {
                "suite": "route1_phase1_5_same_checkpoint_interventions",
                "source": source,
                "runs": [
                    run(
                        pair="tinyllama",
                        seed=42,
                        trained_variant="b2",
                        intervention="b2_eval_k4",
                        output=b2_k4,
                        native_variant="b2",
                        native_csv=b2_native,
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    anomaly_manifest = tmp_path / "anomaly-manifest.json"
    anomaly_manifest.write_text(
        json.dumps(
            {
                "suite": "route1_phase1_5_qwen25_seed44_gate_anomaly",
                "source": source,
                "runs": [
                    run(
                        pair="qwen25_0p5b",
                        seed=44,
                        trained_variant="b6",
                        intervention="b6_gate_alignment_forced_on",
                        output=alignment_forced,
                        native_variant="b6",
                        native_csv=b6_native,
                    ),
                    run(
                        pair="qwen25_0p5b",
                        seed=44,
                        trained_variant="b6",
                        intervention="b6_gate_legacy_forced_on",
                        output=legacy_forced,
                        native_variant="b6",
                        native_csv=b6_native,
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    _runs, main_comparisons, _receiver, _metadata = _load_manifest(main_manifest)
    _runs, merged_comparisons, _receiver, _metadata = _load_manifest(
        main_manifest, anomaly_manifest
    )
    assert len(main_comparisons) == 8
    assert [item.name for item in merged_comparisons[:8]] == [
        item.name for item in main_comparisons
    ]
    assert [item.name for item in merged_comparisons[8:]] == [
        "qwen25_seed44_alignment_forced_on_vs_native",
        "qwen25_seed44_legacy_forced_on_vs_native",
    ]

    summary = generate_diagnostics(
        main_manifest,
        tmp_path / "analysis",
        anomaly_manifest_path=anomaly_manifest,
        bootstrap_samples=100,
        bootstrap_seed=31,
    )
    assert summary["anomaly_manifest"] == str(anomaly_manifest.resolve())
    alignment_task = _find(
        summary["paired_interventions"],
        comparison="qwen25_seed44_alignment_forced_on_vs_native",
        pair="qwen25_0p5b",
        seed=44,
        task="task-a",
    )
    alignment_pooled = _find(
        summary["paired_interventions"],
        comparison="qwen25_seed44_alignment_forced_on_vs_native",
        pair="qwen25_0p5b",
        seed=44,
        task="__pooled__",
    )
    assert alignment_task["delta_accuracy"] == 0.0
    assert alignment_pooled["delta_accuracy"] == 0.0
    legacy_task = _find(
        summary["paired_interventions"],
        comparison="qwen25_seed44_legacy_forced_on_vs_native",
        pair="qwen25_0p5b",
        seed=44,
        task="task-a",
    )
    assert legacy_task["delta_accuracy"] == 0.25
    _find(
        summary["hierarchical_interventions"],
        comparison="qwen25_seed44_legacy_forced_on_vs_native",
        aggregation_level="across_seeds_within_pair",
        pair="qwen25_0p5b",
    )
    alignment_oracle = _find(
        summary["oracle_abstention"],
        method="b6_gate_alignment_forced_on",
        pair="qwen25_0p5b",
        seed=44,
        task="task-a",
    )
    alignment_oracle_pooled = _find(
        summary["oracle_abstention"],
        method="b6_gate_alignment_forced_on",
        pair="qwen25_0p5b",
        seed=44,
        task="__pooled__",
    )
    assert alignment_oracle["oracle_headroom_over_fused"] == 0.25
    assert alignment_oracle_pooled["oracle_headroom_over_fused"] == 0.25
    _find(
        summary["oracle_abstention"],
        method="b6_gate_legacy_forced_on",
        pair="qwen25_0p5b",
        seed=44,
        task="__pooled__",
    )
