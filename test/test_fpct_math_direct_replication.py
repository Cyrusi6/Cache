from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from script.analysis.fpct_math_direct_replication_diagnostics import gradient_comparison
from script.analysis.fpct_math_direct_replication_results import SEEDS, decide
from script.experiment.fpct_math_direct_replication_configs import generate


ROOT = Path(__file__).resolve().parents[1]


def test_generated_replication_configs_preserve_recipe_and_fix_mmlu_subjects(tmp_path):
    paths = generate(ROOT, tmp_path)
    assert len(paths) == 28
    for seed in SEEDS[1:]:
        for arm in ("c_post", "f"):
            value = json.loads((tmp_path / f"train_{seed}_{arm}.json").read_text())
            assert value["training"]["seed"] == seed
            assert value["model"]["projector_init_seed"] == seed
            assert value["training"]["expected_optimizer_steps"] == 64
            assert value["training"]["fpct_expected_training_examples"] == 2048
            assert value["model"]["fpct_operator"] == arm
            assert value["model"]["fpct_position_mode"] == "math"
            assert value["training"]["fpct_alignment_cache_path"].endswith(
                "mmlu_auxiliary_train_2048_certified.pt"
            )
        for cell in ("Y_CC", "Y_CF", "Y_FC", "Y_FF"):
            value = yaml.safe_load(
                (tmp_path / f"eval_{seed}_{cell}_mmlu-redux.yaml").read_text()
            )
            assert len(value["eval"]["subjects"]) == 49
            assert value["model"]["rosetta_config"]["fpct_position_mode"] == "math"


def test_gradient_comparison_exact_cosine_and_layer_groups():
    c_post = {
        "projector_list.0.key_in.weight": torch.tensor([1.0, 0.0]),
        "projector_list.0.value_in.weight": torch.tensor([0.0, 2.0]),
        "projector_list.0.key_gate_logit": torch.tensor([1.0]),
    }
    f_grad = {
        "projector_list.0.key_in.weight": torch.tensor([1.0, 0.0]),
        "projector_list.0.value_in.weight": torch.tensor([0.0, -2.0]),
        "projector_list.0.key_gate_logit": torch.tensor([2.0]),
    }
    value = gradient_comparison(c_post, f_grad)
    groups = {row["group"]: row for row in value["layer_groups"]}
    assert groups["key_path"]["cosine"] == 1.0
    assert groups["value_path"]["cosine"] == -1.0
    assert groups["nuisance_gate_confidence"]["norm_ratio_f_over_c_post"] == 2.0
    assert value["global"]["c_post_missing_parameter_count"] == 0


def _seed(seed: int, t: float, o: float, task_t: dict[str, float]):
    return {
        "seed": seed,
        "estimands_pp": {"T": t, "O": o},
        "task_T_pp": task_t,
        "matched_integrity": {"status": "GO"},
    }


def test_replication_decision_fixed_sequence():
    tasks = {"ai2-arc": 1.0, "openbookqa": 1.0, "mmlu-redux": 1.0}
    query = [_seed(seed, 1.2, 0.2, tasks) for seed in SEEDS]
    assert decide(query)["classification"] == "QUERY_TIME_FPCT_CONTINUE"

    training_only = [
        _seed(SEEDS[0], 1.5, -0.2, tasks),
        _seed(SEEDS[1], 1.0, 0.1, tasks),
        _seed(SEEDS[2], 0.8, -0.1, tasks),
    ]
    assert (
        decide(training_only)["classification"]
        == "FACTORIZED_TRAINING_COLLAPSED_INFERENCE"
    )

    stopped = [
        _seed(SEEDS[0], 0.5, 0.2, tasks),
        _seed(SEEDS[1], -0.1, 0.2, tasks),
        _seed(SEEDS[2], 0.4, 0.2, tasks),
    ]
    assert decide(stopped)["classification"] == "STOP_CURRENT_OPERATOR"
