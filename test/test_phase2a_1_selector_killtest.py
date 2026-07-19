from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pytest

from script.analysis import phase2a_1_selector_killtest as selector


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_MANIFEST = (
    REPO_ROOT / "recipe/eval_recipe/phase2a_1/candidate_manifest.json"
)
FEATURE_MANIFEST = (
    REPO_ROOT / "recipe/eval_recipe/phase2a_1/feature_whitelist.json"
)


def _outcomes(utility: int, neutral_correct: int = 1) -> tuple[int, int]:
    if utility == -1:
        return 1, 0
    if utility == 1:
        return 0, 1
    if utility == 0:
        return neutral_correct, neutral_correct
    raise ValueError(utility)


def _observation(
    *,
    pair: str,
    seed: int,
    task: str,
    row: int,
    split: str,
    utility: int,
    content_hash: str | None = None,
    features: Sequence[float] | None = None,
    neutral_correct: int = 1,
) -> selector.Observation:
    receiver_correct, fused_correct = _outcomes(utility, neutral_correct)
    if features is None:
        features = (
            100.0 + row,
            1.0 + 0.01 * row,
            float(1 + row % 4),
            float(row % 7) / 10.0,
            float(row % 3) / 100.0,
        )
    return selector.Observation(
        pair=pair,
        seed=seed,
        task=task,
        subject="synthetic",
        question_id=f"{task}-{row}",
        content_hash=content_hash or f"{task}-content-{row}",
        split=split,
        features=tuple(map(float, features)),
        receiver_correct=receiver_correct,
        fused_correct=fused_correct,
    )


def _candidate_observations(split: str) -> list[selector.Observation]:
    observations: list[selector.Observation] = []
    for pair_index, pair in enumerate(("pair_a", "pair_b")):
        for seed in (42, 43):
            for task_index, task in enumerate(("task_a", "task_b")):
                for row in range(18):
                    utility = (-1, 0, 1)[row % 3]
                    observations.append(
                        _observation(
                            pair=pair,
                            seed=seed,
                            task=task,
                            row=row,
                            split=split,
                            utility=utility,
                            content_hash=f"{task}-group-{row}",
                            features=(
                                100.0 + row + task_index,
                                1.0 + 0.1 * pair_index + 0.01 * row,
                                float(1 + row % 4),
                                float(row % 7) / 10.0,
                                0.0 if pair_index else float(row % 2) / 100.0,
                            ),
                            neutral_correct=(row // 3) % 2,
                        )
                    )
    return observations


def _event_grid(
    *, pairs: Iterable[str], seeds: Iterable[int], tasks: Iterable[str]
) -> tuple[list[selector.Observation], np.ndarray, np.ndarray]:
    observations: list[selector.Observation] = []
    selected_actions: list[bool] = []
    random_actions: list[bool] = []
    utilities = (0, 1, -1, 0)
    neutral_correct = (1, 1, 1, 0)
    selected = (False, True, False, True)
    random = (True, False, True, False)
    for pair in pairs:
        for seed in seeds:
            for task in tasks:
                for row, utility in enumerate(utilities):
                    observations.append(
                        _observation(
                            pair=pair,
                            seed=seed,
                            task=task,
                            row=row,
                            split="test",
                            utility=utility,
                            neutral_correct=neutral_correct[row],
                            content_hash=f"{task}-event-{row}",
                        )
                    )
                    selected_actions.append(selected[row])
                    random_actions.append(random[row])
    return (
        observations,
        np.asarray(selected_actions, dtype=bool),
        np.asarray(random_actions, dtype=bool),
    )


def test_balanced_weights_have_mean_one_and_equal_cell_mass() -> None:
    observations: list[selector.Observation] = []
    cell_sizes = {
        (pair, seed, task): 2 + pair_index + seed_index + task_index
        for pair_index, pair in enumerate(("pair_a", "pair_b"))
        for seed_index, seed in enumerate((42, 43))
        for task_index, task in enumerate(("task_a", "task_b"))
    }
    for (pair, seed, task), size in cell_sizes.items():
        for row in range(size):
            observations.append(
                _observation(
                    pair=pair,
                    seed=seed,
                    task=task,
                    row=row,
                    split="fit",
                    utility=(-1, 0, 1)[row % 3],
                )
            )

    weights = selector._balanced_weights(observations)

    assert weights.dtype == np.float64
    assert weights.mean() == pytest.approx(1.0, abs=1e-12)
    expected_mass = len(observations) / len(cell_sizes)
    for cell in cell_sizes:
        mask = np.asarray(
            [(item.pair, item.seed, item.task) == cell for item in observations]
        )
        assert weights[mask].sum() == pytest.approx(expected_mass, abs=1e-12)


def test_all_twelve_frozen_candidates_fit_and_calibrate_on_synthetic_data() -> None:
    candidate_manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    feature_manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    candidates = candidate_manifest["candidates"]
    feature_order = feature_manifest["feature_order"]
    assert len(candidates) == 12

    fit = _candidate_observations("fit")
    calibration = _candidate_observations("calibration")
    model_selection = _candidate_observations("model_selection")
    x_selection, *_ = selector._arrays(model_selection)

    for candidate in candidates:
        summary, model, base = selector._fit_candidate(
            candidate,
            feature_order,
            fit,
            calibration,
            model_selection,
            threshold_tolerance=1e-12,
        )
        probabilities = selector._predict_probabilities(model, x_selection)
        scores = selector._score(probabilities)

        assert summary["candidate_id"] == candidate["id"]
        assert np.array_equal(model.classes_, selector.CLASS_ORDER)
        assert np.array_equal(base.classes_, selector.CLASS_ORDER)
        assert probabilities.shape == (len(model_selection), 3)
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-10)
        assert np.array_equal(scores, probabilities[:, 2] - probabilities[:, 0])
        assert summary["threshold"]["kind"] in {
            "finite",
            "always_fused",
            "always_receiver",
        }


def test_threshold_tie_prefers_larger_threshold_and_roundtrips() -> None:
    observations = [
        _observation(
            pair="pair",
            seed=42,
            task="task",
            row=0,
            split="calibration",
            utility=1,
        ),
        _observation(
            pair="pair",
            seed=42,
            task="task",
            row=1,
            split="calibration",
            utility=0,
        ),
        _observation(
            pair="pair",
            seed=42,
            task="task",
            row=2,
            split="calibration",
            utility=-1,
        ),
    ]
    scores = np.asarray([0.8, 0.5, 0.2], dtype=np.float64)
    selected = selector._select_threshold(
        observations,
        scores,
        selector._balanced_weights(observations),
        tolerance=1e-12,
    )

    record = selected["threshold"]
    assert record["kind"] == "finite"
    assert selector._threshold_value(record) == 0.5
    assert selected["calibration_selector_accuracy"] == pytest.approx(1.0)
    assert selected["calibration_transfer_rate"] == pytest.approx(1.0 / 3.0)
    assert not bool(scores[1] > selector._threshold_value(record))

    finite = selector._serialize_threshold(0.125)
    assert selector._threshold_value(finite) == 0.125
    assert float.fromhex(finite["float_hex"]) == 0.125
    for value, kind in (
        (-math.inf, "always_fused"),
        (math.inf, "always_receiver"),
    ):
        serialized = selector._serialize_threshold(value)
        assert serialized == {"kind": kind}
        assert selector._threshold_value(serialized) == value


class _ReorderedProbabilityModel:
    classes_ = np.asarray([1, -1, 0], dtype=np.int64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.tile(np.asarray([[0.6, 0.3, 0.1]]), (len(x), 1))


class _MissingClassProbabilityModel:
    classes_ = np.asarray([-1, 1], dtype=np.int64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.tile(np.asarray([[0.4, 0.6]]), (len(x), 1))


def test_probability_columns_are_reordered_and_invalid_values_are_rejected() -> None:
    x = np.zeros((2, 5), dtype=np.float64)
    probabilities = selector._predict_probabilities(_ReorderedProbabilityModel(), x)
    assert np.array_equal(
        probabilities,
        np.asarray([[0.3, 0.1, 0.6], [0.3, 0.1, 0.6]]),
    )

    with pytest.raises(ValueError, match="classes are invalid"):
        selector._predict_probabilities(_MissingClassProbabilityModel(), x)

    invalid = (
        np.asarray([[math.nan, 0.5, 0.5]]),
        np.asarray([[-0.1, 0.5, 0.6]]),
        np.asarray([[0.2, 0.2, 0.2]]),
        np.asarray([[0.5, 0.5]]),
    )
    for values in invalid:
        with pytest.raises(ValueError):
            selector._validate_probabilities(values)


def test_same_rate_random_is_deterministic_and_constant_within_content_group() -> None:
    observations: list[selector.Observation] = []
    selector_actions: list[bool] = []
    for pair in ("pair_a", "pair_b"):
        for task in ("task_a", "task_b"):
            for seed in (42, 43):
                for group, member_count in (("g0", 1), ("g1", 2), ("g2", 1)):
                    for member in range(member_count):
                        row = len(observations)
                        observations.append(
                            _observation(
                                pair=pair,
                                seed=seed,
                                task=task,
                                row=row,
                                split="test",
                                utility=(-1, 0, 1)[member % 3],
                                content_hash=f"{task}-{group}",
                            )
                        )
                        selector_actions.append((row + seed) % 3 == 0)
    protocol = {"random_baseline": {"seed": 20260721}}

    first_actions, first_audit = selector._same_rate_random_actions(
        observations, np.asarray(selector_actions, dtype=bool), protocol
    )
    second_actions, second_audit = selector._same_rate_random_actions(
        observations, np.asarray(selector_actions, dtype=bool), protocol
    )

    assert np.array_equal(first_actions, second_actions)
    assert first_audit == second_audit
    groups: dict[tuple[str, str, str], list[bool]] = {}
    for item, action in zip(observations, first_actions):
        groups.setdefault((item.pair, item.task, item.content_hash), []).append(
            bool(action)
        )
    assert groups
    assert all(len(set(actions)) == 1 for actions in groups.values())
    for audit in first_audit.values():
        assert audit["absolute_rate_gap"] == pytest.approx(
            abs(audit["achieved_transferred_rows"] - audit["target_transferred_rows"])
            / audit["total_rows"]
        )


def test_bootstrap_is_deterministic_and_metric_formulas_match_manual_values() -> None:
    pairs = ("pair_a", "pair_b")
    seeds = (42, 43)
    tasks = ("task_a", "task_b")
    observations, selected_actions, random_actions = _event_grid(
        pairs=pairs, seeds=seeds, tasks=tasks
    )
    probabilities = np.tile(
        np.asarray(
            [
                [0.1, 0.8, 0.1],
                [0.05, 0.05, 0.9],
                [0.9, 0.05, 0.05],
                [0.1, 0.8, 0.1],
            ],
            dtype=np.float64,
        ),
        (len(observations) // 4, 1),
    )

    metrics = selector._metric_values(
        observations,
        selected_actions,
        comparator_policy="always_fused",
        weighting="task_macro",
        probabilities=probabilities,
    )
    assert metrics["receiver_accuracy"] == pytest.approx(0.5)
    assert metrics["fused_accuracy"] == pytest.approx(0.5)
    assert metrics["selector_accuracy"] == pytest.approx(0.75)
    assert metrics["selector_minus_comparator"] == pytest.approx(0.25)
    assert metrics["oracle_accuracy"] == pytest.approx(0.75)
    assert metrics["oracle_headroom_recovery"] == pytest.approx(1.0)
    assert metrics["transfer_rate"] == pytest.approx(0.5)
    assert metrics["abstention_rate"] == pytest.approx(0.5)
    assert metrics["harmful_reduction"] == pytest.approx(1.0)
    assert metrics["beneficial_retention"] == pytest.approx(1.0)
    assert metrics["benefit_auprc_pooled"] == pytest.approx(1.0)
    assert metrics["harm_auprc_pooled"] == pytest.approx(1.0)
    assert metrics["multiclass_brier"] == pytest.approx(0.0375)
    assert metrics["ece_harm"] == pytest.approx(0.0875)
    assert metrics["ece_neutral"] == pytest.approx(0.125)
    assert metrics["ece_benefit"] == pytest.approx(0.0875)
    assert metrics["ece_multiclass_macro"] == pytest.approx(0.1)

    bootstrap_args = dict(
        observations=observations,
        selected_actions=selected_actions,
        random_actions=random_actions,
        comparator_policy="always_fused",
        pair_order=pairs,
        seed_order=seeds,
        task_order=tasks,
        samples=200,
        base_seed=20260719,
        batch_size=37,
        context="synthetic",
    )
    first = selector._build_bootstrap_tensors(**bootstrap_args)
    second = selector._build_bootstrap_tensors(**bootstrap_args)
    for task in tasks:
        for name in ("point_sums", "point_counts", "boot_sums", "boot_counts"):
            assert np.array_equal(
                first["task_tensors"][task][name],
                second["task_tensors"][task][name],
            )

    aggregate_args = dict(
        pairs=pairs,
        seeds=seeds,
        tasks=tasks,
        weighting="task_macro",
        resample_pairs=True,
        resample_seeds=True,
        base_seed=20260719,
        label="synthetic-primary",
    )
    first_point, first_bootstrap = selector._aggregate_tensor_scope(
        first, **aggregate_args
    )
    second_point, second_bootstrap = selector._aggregate_tensor_scope(
        second, **aggregate_args
    )
    assert np.array_equal(first_point, second_point)
    assert np.array_equal(first_bootstrap, second_bootstrap)

    component_metrics = selector._metrics_from_component_rates(
        first_point, "selected"
    )
    for name in (
        "receiver_accuracy",
        "fused_accuracy",
        "selector_accuracy",
        "selector_minus_comparator",
        "oracle_accuracy",
        "oracle_headroom_recovery",
        "transfer_rate",
        "harmful_reduction",
        "beneficial_retention",
    ):
        assert float(component_metrics[name]) == pytest.approx(metrics[name])
