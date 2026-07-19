from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from script.analysis import phase2a_2a_cache_geometry_stats as stats


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "recipe/eval_recipe/phase2a_2a/protocol_manifest.json"
FEATURE_PATH = REPO_ROOT / "recipe/eval_recipe/phase2a_2a/feature_manifest.json"
CANDIDATE_PATH = REPO_ROOT / "recipe/eval_recipe/phase2a_2a/candidate_manifest.json"
SCHEMA_PATH = (
    REPO_ROOT / "recipe/eval_recipe/phase2a_2a/cache_geometry_artifact_schema.json"
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _geometry(
    *, pair: str, task: str, row: int, features: Sequence[float], seed: int = 42
) -> stats.GeometrySample:
    key: stats.IdentityKey = (
        pair,
        seed,
        task,
        "subject",
        str(row),
        _hash(f"{task}/{row}"),
    )
    return stats.GeometrySample(
        key=key,
        features=tuple(map(float, features)),
        within_key_variation=True,
        within_value_variation=True,
    )


def _observation(
    *,
    pair: str,
    task: str,
    row: int,
    utility: int,
    features: Sequence[float],
    fold: int = 0,
    seed: int = 42,
) -> stats.Observation:
    if utility == -1:
        receiver, fused = 1, 0
    elif utility == 1:
        receiver, fused = 0, 1
    elif utility == 0:
        receiver = fused = row % 2
    else:  # pragma: no cover - helper contract
        raise ValueError(utility)
    return stats.Observation(
        geometry=_geometry(
            pair=pair, task=task, row=row, features=features, seed=seed
        ),
        receiver_correct=receiver,
        fused_correct=fused,
        fold=fold,
    )


def _minimal_protocol() -> dict[str, Any]:
    return {
        "source": {"dataset_content_sha256": "d" * 64},
        "scope": {
            "primary_pairs": ["pair_a", "pair_b", "pair_c"],
            "seeds": [42],
            "tasks": ["task"],
        },
        "fold": {
            "count": 5,
            "outer_prefix": "phase2a2a-outer-v1",
            "development": {
                "prefix": "phase2a2a-dev-v1",
                "fit_interval": [0.0, 0.6],
                "calibration_interval": [0.6, 0.8],
                "model_selection_interval": [0.8, 1.0],
            },
        },
        "calibration_and_selection": {"threshold_tolerance": 1e-12},
        "canonical_output_fingerprint": {
            "matched_runtime_control_scope": {
                "pair": "pair_a",
                "seed": 42,
                "task": "task",
                "expected_rows": 2,
            }
        },
        "go_gate": {
            "gates_in_order": [
                "on_off_output_exact",
                "real_geometry_variation_nonconstant",
                "pooled_harm_auprc_margin",
                "held_out_pair_harm_auprc",
                "selector_accuracy_gain",
                "harmful_reduction",
                "beneficial_retention",
                "every_pair_noninferiority",
                "brier_beats_crossfit_prior",
            ],
            "pooled_harm_auprc_minimum_margin_over_prevalence": 0.03,
            "minimum_pairs_harm_auprc_above_own_prevalence": 2,
            "minimum_selector_minus_always_fused": 0.005,
            "minimum_harmful_reduction": 0.15,
            "minimum_beneficial_retention": 0.9,
            "minimum_each_pair_selector_minus_fused": -0.002,
        },
    }


def test_frozen_feature_candidate_and_protocol_contracts_are_explicit() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    features = json.loads(FEATURE_PATH.read_text(encoding="utf-8"))
    candidates = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))

    assert protocol["scope"]["seeds"] == [42]
    assert protocol["fold"]["outer_prefix"] == "phase2a2a-outer-v1"
    assert protocol["fold"]["development"]["fit_interval"] == [0.0, 0.6]
    assert protocol["fold"]["development"]["calibration_interval"] == [0.6, 0.8]
    assert protocol["fold"]["development"]["model_selection_interval"] == [0.8, 1.0]
    assert len(protocol["go_gate"]["gates_in_order"]) == 9
    assert features["feature_count"] == 177 == len(features["feature_order"])
    assert len(set(features["feature_order"])) == 177
    assert not any("*" in name for name in features["feature_order"])
    assert candidates["candidate_count"] == 184
    for family in (
        "single_feature_stumps",
        "l2_multinomial_logistic",
        "depth2_trees",
    ):
        assert candidates["candidates"][family]["explicit_features"] == features[
            "feature_order"
        ]
    loaded_protocol, loaded_features, loaded_candidates = stats._verify_design_bundle(
        PROTOCOL_PATH, FEATURE_PATH, CANDIDATE_PATH, SCHEMA_PATH
    )
    assert loaded_protocol == protocol
    assert loaded_features == features
    assert loaded_candidates == candidates


def test_layer_records_aggregate_to_frozen_compact_features(tmp_path: Path) -> None:
    feature_manifest = json.loads(FEATURE_PATH.read_text(encoding="utf-8"))
    source_mapping = feature_manifest["per_layer_source_mapping"]
    path = tmp_path / "layers.jsonl"
    content_hash = _hash("content")
    rows = []
    for layer in range(3):
        row: dict[str, Any] = {
            "cache_geometry_schema_version": 1,
            "role": "geometry_on",
            "pair": "pair",
            "seed": 42,
            "task": "task",
            "subject": "subject",
            "question_id": "0",
            "content_hash": content_hash,
            "projector_index": layer,
            "target_layer": layer,
            "batch_index": 0,
        }
        for base, source in source_mapping.items():
            if str(source).startswith("derived:"):
                continue
            if base in stats.SCALAR_FEATURES:
                row[source] = {
                    "source_receiver_length_ratio": 0.5,
                    "valid_alignment_mass": 1.0,
                    "valid_alignment_coverage": 1.0,
                }[base]
            else:
                row[source] = float(layer + 1)
        row["key_residual_to_native_norm_ratio"] = float(layer + 1)
        row["value_residual_to_native_norm_ratio"] = 1.0
        rows.append(row)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    samples = stats._aggregate_geometry([path], feature_manifest)
    sample = next(iter(samples.values()))
    values = dict(zip(feature_manifest["feature_order"], sample.features))

    assert values["key_native_norm__all_mean"] == pytest.approx(2.0)
    assert values["key_native_norm__all_std"] == pytest.approx(np.std([1, 2, 3]))
    assert values["key_native_norm__early_mean"] == pytest.approx(1.0)
    assert values["key_native_norm__late_mean"] == pytest.approx(3.0)
    assert values["residual_imbalance__all_mean"] == pytest.approx(
        np.mean(np.log([1.0, 2.0, 3.0]))
    )
    assert values["source_receiver_length_ratio"] == pytest.approx(0.5)
    assert sample.within_key_variation
    assert sample.within_value_variation


def test_matched_runtime_parity_only_requires_frozen_tiny_arc_scope() -> None:
    protocol = _minimal_protocol()
    keys = [
        ("pair_a", 42, "task", "s", "0", _hash("0")),
        ("pair_a", 42, "task", "s", "1", _hash("1")),
        ("pair_b", 42, "task", "s", "0", _hash("0")),
    ]
    on = {key: _hash(f"on/{index}") for index, key in enumerate(keys)}
    off = {key: on[key] for key in keys[:2]}

    exact = stats._output_parity(on, off, protocol)
    assert exact["exact"]
    assert exact["off_count"] == 2

    off[keys[0]] = _hash("different")
    assert not stats._output_parity(on, off, protocol)["exact"]


def test_outcome_identity_scan_does_not_parse_correctness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "outcomes.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=stats.OUTCOME_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "pair": "pair",
                "seed": 42,
                "task": "task",
                "subject": "subject",
                "question_id": "0",
                "content_hash": _hash("content"),
                "receiver_correct": "NOT_PARSED",
                "fused_correct": "ALSO_NOT_PARSED",
            }
        )

    def forbidden(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("freeze identity scan must not parse correctness")

    monkeypatch.setattr(stats, "_strict_bool", forbidden)
    identities = stats._load_outcomes(path, parse_values=False)
    assert len(identities) == 1
    assert next(iter(identities.values())) is None


def test_equivalence_report_requires_nine_cells_and_one_matched_control(
    tmp_path: Path,
) -> None:
    pairs = ["tinyllama", "qwen25_0p5b", "llama32_1b"]
    tasks = ["ai2-arc", "openbookqa", "mmlu-redux"]
    exact_columns = [
        "pred",
        "is_correct",
        "cot_pred",
        "cot_output",
        "cot_gen_length",
    ]
    runs = []
    comparisons = []
    for pair in pairs:
        for task in tasks:
            run_id = f"{pair}__seed42__{task}__instrumented"
            reference = tmp_path / f"{run_id}.reference.csv"
            prediction = tmp_path / f"{run_id}.prediction.csv"
            reference.write_text("frozen-reference\n", encoding="utf-8")
            prediction.write_text("instrumented\n", encoding="utf-8")
            runs.append(
                {
                    "id": run_id,
                    "pair": pair,
                    "seed": 42,
                    "dataset": task,
                    "kind": "instrumented",
                    "reference_prediction": stats._path_record(reference),
                }
            )
            comparisons.append(
                {
                    "run_id": run_id,
                    "kind": "instrumented",
                    "rows": 1,
                    "expected_rows": 1,
                    "keys_exact": True,
                    "exact_columns": exact_columns,
                    "mismatch_count_capped": 0,
                    "exact": True,
                    "prediction_path": str(prediction),
                    "prediction_sha256": stats._sha256(prediction),
                }
            )
    control_id = "tinyllama__seed42__ai2-arc__instrumentation_off"
    control_reference = tmp_path / "control.reference.csv"
    control_prediction = tmp_path / "control.prediction.csv"
    control_reference.write_text("frozen-reference\n", encoding="utf-8")
    control_prediction.write_text("control\n", encoding="utf-8")
    runs.append(
        {
            "id": control_id,
            "pair": "tinyllama",
            "seed": 42,
            "dataset": "ai2-arc",
            "kind": "overhead_control",
            "reference_prediction": stats._path_record(control_reference),
        }
    )
    comparisons.append(
        {
            "run_id": control_id,
            "kind": "overhead_control",
            "rows": 1,
            "expected_rows": 1,
            "keys_exact": True,
            "exact_columns": exact_columns,
            "mismatch_count_capped": 0,
            "exact": True,
            "prediction_path": str(control_prediction),
            "prediction_sha256": stats._sha256(control_prediction),
        }
    )
    execution_path = tmp_path / "execution.json"
    report_path = tmp_path / "report.json"
    execution_path.write_text(
        json.dumps(
            {
                "role": "pre_transfer_cache_geometry_pilot_execution_manifest",
                "constraints": {"allowed_seed": [42], "allowed_split": "fit"},
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(
            {
                "phase": "Phase 2A-2a",
                "instrumentation_output_exact": True,
                "comparisons": comparisons,
            }
        ),
        encoding="utf-8",
    )
    protocol = {
        "reference_equivalence": {
            "execution_manifest_role": "pre_transfer_cache_geometry_pilot_execution_manifest",
            "verify_report_phase": "Phase 2A-2a",
            "instrumented_cell_count": 9,
            "exact_columns": exact_columns,
        },
        "scope": {"primary_pairs": pairs, "tasks": tasks},
        "canonical_output_fingerprint": {
            "matched_runtime_control_scope": {
                "pair": "tinyllama",
                "seed": 42,
                "task": "ai2-arc",
            }
        },
        "source": {"expected_fit_rows_by_task": {task: 1 for task in tasks}},
    }

    audit = stats._validate_equivalence_contract(
        execution_path, report_path, protocol
    )
    assert audit["instrumented_comparison_count"] == 9
    assert audit["matched_off_control_count"] == 1
    assert audit["all_exact"]

    comparisons[0]["exact"] = False
    report_path.write_text(
        json.dumps(
            {
                "phase": "Phase 2A-2a",
                "instrumentation_output_exact": True,
                "comparisons": comparisons,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="failed contract"):
        stats._validate_equivalence_contract(execution_path, report_path, protocol)


def test_pair_task_weights_equalize_cells_but_pool_seeds() -> None:
    observations = []
    row = 0
    for pair, task, seed, count in (
        ("a", "x", 42, 2),
        ("a", "x", 43, 3),
        ("a", "y", 42, 7),
        ("b", "x", 42, 4),
        ("b", "y", 42, 9),
    ):
        for _ in range(count):
            observations.append(
                _observation(
                    pair=pair,
                    task=task,
                    row=row,
                    utility=(-1, 0, 1)[row % 3],
                    features=[float(row)],
                    seed=seed,
                )
            )
            row += 1
    weights = stats._pair_task_weights(observations)
    masses: dict[tuple[str, str], float] = {}
    for item, weight in zip(observations, weights):
        cell = (item.pair, item.task)
        masses[cell] = masses.get(cell, 0.0) + float(weight)
    assert weights.mean() == pytest.approx(1.0)
    assert len(set(round(value, 12) for value in masses.values())) == 1
    a_x = [
        weight
        for item, weight in zip(observations, weights)
        if (item.pair, item.task) == ("a", "x")
    ]
    assert len(set(round(float(value), 12) for value in a_x)) == 1


def test_frozen_candidate_families_fit_calibrate_and_select() -> None:
    feature_order = ["signal", "noise"]
    candidates = [
        {
            "id": "stump",
            "ordinal": 0,
            "family": "single_feature_stump",
            "features": ["signal"],
            "params": {
                "criterion": "gini",
                "splitter": "best",
                "max_depth": 1,
                "min_weight_fraction_leaf": 0.01,
                "class_weight": None,
                "random_state": 7,
            },
        },
        {
            "id": "logreg",
            "ordinal": 1,
            "family": "l2_multinomial_logistic",
            "features": feature_order,
            "params": {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "tol": 1e-10,
                "max_iter": 5000,
                "fit_intercept": True,
                "class_weight": None,
                "random_state": 7,
            },
        },
        {
            "id": "tree",
            "ordinal": 2,
            "family": "shallow_decision_tree",
            "features": feature_order,
            "params": {
                "criterion": "gini",
                "splitter": "best",
                "max_depth": 2,
                "min_weight_fraction_leaf": 0.01,
                "class_weight": None,
                "random_state": 7,
            },
        },
    ]

    def split_rows(offset: int) -> list[stats.Observation]:
        rows = []
        row_id = offset * 1000
        for pair in ("a", "b"):
            for task in ("x", "y"):
                for repeat in range(8):
                    for utility in (-1, 0, 1):
                        signal = float(utility) + 0.01 * repeat
                        rows.append(
                            _observation(
                                pair=pair,
                                task=task,
                                row=row_id,
                                utility=utility,
                                features=[signal, float((row_id * 7) % 11)],
                            )
                        )
                        row_id += 1
        return rows

    fit = split_rows(1)
    calibration = split_rows(2)
    selection = split_rows(3)
    candidate, model, threshold, summaries = stats._fit_select_candidate(
        candidates,
        feature_order,
        fit,
        calibration,
        selection,
        1e-12,
    )
    probabilities = stats._predict_probabilities(model, stats._arrays(selection)[0])

    assert candidate["id"] in {"stump", "logreg", "tree"}
    assert len(summaries) == 3
    assert probabilities.shape == (len(selection), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.array_equal(stats._score(probabilities), probabilities[:, 2] - probabilities[:, 0])
    assert np.isfinite(threshold) or threshold in {-np.inf, np.inf}


def test_crossfit_excludes_held_pair_and_outer_content_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _minimal_protocol()
    feature_order = ["signal"]
    candidate_manifest = {
        "candidate_count": 1,
        "candidates": {
            "single_feature_stumps": {
                "explicit_features": feature_order,
                "params": {
                    "criterion": "gini",
                    "splitter": "best",
                    "max_depth": 1,
                    "min_weight_fraction_leaf": 0.01,
                    "class_weight": None,
                    "random_state": 1,
                },
            },
            "l2_multinomial_logistic": {
                "explicit_features": feature_order,
                "C": [],
                "params": {},
            },
            "depth2_trees": {
                "explicit_features": feature_order,
                "min_weight_fraction_leaf": [],
                "params": {},
            },
        },
    }

    hashes_by_fold_role: dict[tuple[int, str], str] = {}
    probe = 0
    while len(hashes_by_fold_role) < 15:
        content_hash = _hash(f"probe/{probe}")
        key = (
            stats._content_fold(content_hash, protocol),
            stats._development_role(content_hash, protocol),
        )
        hashes_by_fold_role.setdefault(key, content_hash)
        probe += 1
    observations = []
    row = 0
    for pair in protocol["scope"]["primary_pairs"]:
        for (fold, role), content_hash in sorted(hashes_by_fold_role.items()):
            for utility in (-1, 0, 1):
                geometry = stats.GeometrySample(
                    key=(pair, 42, "task", "s", str(row), content_hash),
                    features=(float(utility),),
                    within_key_variation=True,
                    within_value_variation=True,
                )
                receiver, fused = ({-1: (1, 0), 0: (1, 1), 1: (0, 1)})[
                    utility
                ]
                observations.append(
                    stats.Observation(geometry, receiver, fused, fold)
                )
                row += 1

    captured: list[tuple[set[str], set[int], set[str]]] = []

    class DummyModel:
        classes_ = stats.CLASS_ORDER

    def fake_fit_select(
        candidates: Sequence[Mapping[str, Any]],
        _feature_order: Sequence[str],
        fit: Sequence[stats.Observation],
        calibration: Sequence[stats.Observation],
        selection: Sequence[stats.Observation],
        _tolerance: float,
    ) -> tuple[Mapping[str, Any], DummyModel, float, list[dict[str, Any]]]:
        development = [*fit, *calibration, *selection]
        captured.append(
            (
                {item.pair for item in development},
                {item.fold for item in development},
                {stats._development_role(item.content_hash, protocol) for item in fit},
            )
        )
        return candidates[0], DummyModel(), 0.0, []

    monkeypatch.setattr(stats, "_fit_select_candidate", fake_fit_select)
    monkeypatch.setattr(
        stats,
        "_predict_probabilities",
        lambda _model, x: np.tile(np.asarray([[0.2, 0.3, 0.5]]), (len(x), 1)),
    )
    predictions, audits = stats._crossfit(
        observations, feature_order, candidate_manifest, protocol
    )

    assert len(predictions) == len(observations)
    assert len(audits) == 15
    for audit, (pairs, folds, fit_roles) in zip(audits, captured):
        assert audit["held_out_pair"] not in pairs
        assert audit["evaluation_fold"] not in folds
        assert fit_roles == {"fit"}
        assert audit["content_overlap_count"] == 0


def test_nine_go_gates_use_frozen_inclusive_and_strict_boundaries() -> None:
    protocol = _minimal_protocol()
    pooled = {
        "harm_auprc": 0.23,
        "harm_prevalence": 0.20,
        "selector_minus_fused": 0.005,
        "harmful_reduction": 0.15,
        "beneficial_retention": 0.90,
        "multiclass_brier": 0.49,
        "crossfit_prior_brier": 0.50,
    }
    pair_metrics = {
        "a": {"harm_auprc": 0.21, "harm_prevalence": 0.20, "selector_minus_fused": -0.002},
        "b": {"harm_auprc": 0.31, "harm_prevalence": 0.30, "selector_minus_fused": 0.0},
        "c": {"harm_auprc": 0.10, "harm_prevalence": 0.10, "selector_minus_fused": 0.01},
    }
    result = stats.evaluate_go_gates(
        pooled_metrics=pooled,
        pair_metrics=pair_metrics,
        reference_equivalence={"all_exact": True},
        output_parity={"exact": True},
        geometry_audit={"all_pairs_passed": True},
        protocol=protocol,
    )
    assert result["decision"] == "GO"
    assert len(result["gates"]) == 9
    assert all(result["gates"].values())

    pooled_equal_brier = {**pooled, "multiclass_brier": 0.50}
    failed = stats.evaluate_go_gates(
        pooled_metrics=pooled_equal_brier,
        pair_metrics=pair_metrics,
        reference_equivalence={"all_exact": True},
        output_parity={"exact": True},
        geometry_audit={"all_pairs_passed": True},
        protocol=protocol,
    )
    assert failed["decision"] == "NO_GO"
    assert not failed["gates"]["brier_beats_crossfit_prior"]
