from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml

from script.analysis import route1_identifiability_suite as suite


def _generate(
    tmp_path: Path,
    reuse_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], Path]:
    output_root = tmp_path / "suite"
    manifest = suite.generate_suite(
        template_path=suite.DEFAULT_TEMPLATE,
        output_root=output_root,
        repo_root=suite.REPO_ROOT,
        reuse_overrides=reuse_overrides,
    )
    return manifest, output_root


def _run(
    manifest: dict[str, Any], pair: str, variant: str, seed: int
) -> dict[str, Any]:
    return next(
        run
        for run in manifest["runs"]
        if run["pair"] == pair and run["variant"] == variant and run["seed"] == seed
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _write_eval_config(
    path: Path, dataset: str, gpu_ids: list[int], output: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"model_name": "Qwen/Qwen3-0.6B"},
                "output": {"output_dir": str(output)},
                "eval": {"dataset": dataset, "gpu_ids": gpu_ids},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _make_eval_bundle(
    tmp_path: Path, name: str
) -> tuple[dict[str, str], dict[str, str]]:
    configs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    for dataset, gpu_ids in {
        "ai2-arc": [0],
        "openbookqa": [1],
        "mmlu-redux": [2, 3],
    }.items():
        output = tmp_path / "results" / name / dataset
        config = tmp_path / "configs" / name / f"{dataset}.yaml"
        _write_eval_config(config, dataset, gpu_ids, output)
        configs[dataset] = str(config)
        outputs[dataset] = str(output)
    return configs, outputs


def _complete_eval_outputs(
    output_dirs: dict[str, str], *, gate_required: bool = False
) -> None:
    for dataset, output in output_dirs.items():
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        expected_rows = int(suite.EVAL_LAYOUT[dataset]["expected_rows"])
        prediction_rows = ["subject,question_id,true_answer,pred,is_correct\n"]
        prediction_rows.extend(
            f"main,{index},A,A,true\n" for index in range(expected_rows)
        )
        (path / "run_cot.csv").write_text("".join(prediction_rows), encoding="utf-8")
        length_group = str(suite.EVAL_LAYOUT[dataset]["length_group"])
        (path / "run_summary.json").write_text(
            json.dumps(
                {
                    "model": "Rosetta",
                    "dataset": dataset,
                    "answer_method": "generate",
                    "overall_accuracy": 1.0,
                    "subjects": {"main": 1.0},
                    "length_statistics": {
                        length_group: {
                            "main": {
                                "accuracy": 1.0,
                                "total_samples": expected_rows,
                            }
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (path / "run_gate_diagnostics.json").write_text(
            json.dumps(
                (
                    {
                        "status": "compact",
                        "counts": {
                            "examples_seen": expected_rows,
                            "examples_with_gate": expected_rows,
                        },
                    }
                    if gate_required
                    else {"status": "unavailable"}
                )
            )
            + "\n",
            encoding="utf-8",
        )


def _complete_checkpoint(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    layer_map = {
        str(index): [[max(0, index - 6), index]]
        for index in range(suite.EXPECTED_PROJECTOR_COUNT)
    }
    (path / "projector_config.json").write_text(
        json.dumps({"0": {"1": layer_map}}) + "\n", encoding="utf-8"
    )
    for index in range(suite.EXPECTED_PROJECTOR_COUNT):
        torch.save({"weight": torch.ones(1)}, path / f"projector_{index}.pt")
        (path / f"projector_{index}.json").write_text(
            json.dumps({"class": "C2CProjector", "init_args": {"index": index}})
            + "\n",
            encoding="utf-8",
        )


def _complete_posthoc_gate_diagnostics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "counts": {
                    "examples_seen": 64,
                    "examples_with_gate": 64,
                    "token_head_gate_projectors": 28,
                },
                "metadata": {"processed_samples": 64},
                "by_layer": [{"layer": layer} for layer in range(28)],
                "by_stage": [{"stage": "early"}],
                "by_layer_head": [{"layer": 0, "head": 0}],
                "by_relative_token_bin": [{"relative_token_bin": 0}],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_posthoc_gate_diagnostics_required_contract(tmp_path: Path) -> None:
    artifact = tmp_path / "gate" / "gate_diagnostics.json"
    required = {"required": True, "artifact": str(artifact), "num_samples": 64}

    assert suite._posthoc_gate_diagnostics_complete({"required": False})
    assert not suite._posthoc_gate_diagnostics_complete(required)

    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "status": "unavailable",
                "counts": {"examples_with_gate": 0},
            }
        ),
        encoding="utf-8",
    )
    assert not suite._posthoc_gate_diagnostics_complete(required)

    _complete_posthoc_gate_diagnostics(artifact)
    assert suite._posthoc_gate_diagnostics_complete(required)
    partial = _load_json(artifact)
    partial["metadata"]["processed_samples"] = 63
    artifact.write_text(json.dumps(partial) + "\n", encoding="utf-8")
    assert not suite._posthoc_gate_diagnostics_complete(required)


def test_checkpoint_completeness_requires_all_readable_projectors(
    tmp_path: Path,
) -> None:
    complete = tmp_path / "complete"
    _complete_checkpoint(complete)
    assert suite._checkpoint_complete(complete)
    directory_sha256 = suite._checkpoint_directory_sha256(complete)
    assert suite._checkpoint_complete(complete, directory_sha256)
    assert not suite._checkpoint_complete(complete, "0" * 64)

    missing = tmp_path / "missing"
    _complete_checkpoint(missing)
    (missing / "projector_27.pt").unlink()
    assert not suite._checkpoint_complete(missing)

    corrupt = tmp_path / "corrupt"
    _complete_checkpoint(corrupt)
    (corrupt / "projector_11.pt").write_bytes(b"not-a-torch-state-dict")
    assert not suite._checkpoint_complete(corrupt)

    wrong_mapping = tmp_path / "wrong-mapping"
    _complete_checkpoint(wrong_mapping)
    config = _load_json(wrong_mapping / "projector_config.json")
    del config["0"]["1"]["27"]
    (wrong_mapping / "projector_config.json").write_text(
        json.dumps(config) + "\n", encoding="utf-8"
    )
    assert not suite._checkpoint_complete(wrong_mapping)

    duplicate_reference = tmp_path / "duplicate-reference"
    _complete_checkpoint(duplicate_reference)
    config = _load_json(duplicate_reference / "projector_config.json")
    config["0"]["1"]["0"].append([0, 0])
    (duplicate_reference / "projector_config.json").write_text(
        json.dumps(config) + "\n", encoding="utf-8"
    )
    assert not suite._checkpoint_complete(duplicate_reference)


def test_evaluation_completeness_validates_unique_artifacts_rows_and_summary(
    tmp_path: Path,
) -> None:
    _configs, complete = _make_eval_bundle(tmp_path, "complete")
    _complete_eval_outputs(complete)
    assert suite._evaluation_complete(complete)
    assert not suite._evaluation_complete(complete, gate_required=True)
    _complete_eval_outputs(complete, gate_required=True)
    assert suite._evaluation_complete(complete, gate_required=True)
    biased_gate = Path(complete["ai2-arc"]) / "run_gate_diagnostics.json"
    biased = _load_json(biased_gate)
    biased["counts"]["examples_with_gate"] -= 1
    biased_gate.write_text(json.dumps(biased) + "\n", encoding="utf-8")
    assert not suite._evaluation_complete(complete, gate_required=True)

    _configs, empty = _make_eval_bundle(tmp_path, "empty")
    _complete_eval_outputs(empty)
    Path(empty["ai2-arc"]).joinpath("run_cot.csv").write_text("", encoding="utf-8")
    assert not suite._evaluation_complete(empty)

    _configs, duplicate = _make_eval_bundle(tmp_path, "duplicate")
    _complete_eval_outputs(duplicate)
    Path(duplicate["openbookqa"]).joinpath("second_summary.json").write_text(
        Path(duplicate["openbookqa"]).joinpath("run_summary.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    assert suite._evaluation_complete(duplicate)

    retry_dir = Path(duplicate["openbookqa"])
    (retry_dir / "retry_cot.csv").write_text("broken\n", encoding="utf-8")
    (retry_dir / "retry_summary.json").write_text("{}\n", encoding="utf-8")
    (retry_dir / "retry_gate_diagnostics.json").write_text(
        json.dumps({"status": "ok"}) + "\n", encoding="utf-8"
    )
    selected = suite._latest_complete_evaluation_attempt(
        "openbookqa", duplicate["openbookqa"]
    )
    assert selected is not None
    assert selected["prediction"].name == "run_cot.csv"
    assert suite._evaluation_complete(duplicate)

    _configs, wrong_dataset = _make_eval_bundle(tmp_path, "wrong-dataset")
    _complete_eval_outputs(wrong_dataset)
    summary_path = Path(wrong_dataset["mmlu-redux"]) / "run_summary.json"
    summary = _load_json(summary_path)
    summary["dataset"] = "ai2-arc"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    assert not suite._evaluation_complete(wrong_dataset)

    _configs, wrong_accuracy = _make_eval_bundle(tmp_path, "wrong-accuracy")
    _complete_eval_outputs(wrong_accuracy)
    summary_path = Path(wrong_accuracy["ai2-arc"]) / "run_summary.json"
    summary = _load_json(summary_path)
    summary["overall_accuracy"] = 0.5
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    assert not suite._evaluation_complete(wrong_accuracy)

    _configs, wrong_rows = _make_eval_bundle(tmp_path, "wrong-rows")
    _complete_eval_outputs(wrong_rows)
    prediction_path = Path(wrong_rows["openbookqa"]) / "run_cot.csv"
    rows = prediction_path.read_text(encoding="utf-8").splitlines()
    prediction_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    assert not suite._evaluation_complete(wrong_rows)


def test_frozen_template_hash_and_semantics() -> None:
    assert suite.DEFAULT_TEMPLATE.is_file()
    assert suite._sha256(suite.DEFAULT_TEMPLATE) == suite.FROZEN_TEMPLATE_SHA256
    frozen = _load_json(suite.DEFAULT_TEMPLATE)
    suite._ensure_template_is_v22(frozen, suite.DEFAULT_TEMPLATE)

    legacy = (
        suite.REPO_ROOT / "local/tmp/train_recipes/route1_alignment_v22/"
        "qwen3_0.6b_tinyllama1.1b_soft_span_overlap_v22_"
        "token_mlp_entropy050_small2048.json"
    )
    if legacy.is_file():
        assert frozen == _load_json(legacy)
    assert suite.DEFAULT_STEP1_REUSE_OVERRIDE.is_file()
    assert (
        _load_json(suite.DEFAULT_STEP1_REUSE_OVERRIDE)
        == suite._current_b6_reuse_override()
    )


def test_generate_suite_builds_matrix_lane_plans_and_analysis_contract(
    tmp_path: Path,
) -> None:
    manifest, output_root = _generate(tmp_path)
    revision = f"rev_{manifest['git_commit'][:12]}"
    assert manifest["revision_namespace"] == revision

    assert manifest["summary"] == {
        "run_count": 67,
        "train_run_count": 66,
        "eval_triplet_count": 67,
        "posthoc_gate_diagnostic_count": 26,
        "conditional_run_count": 30,
        "note": (
            "Recommended commands launch lane runners. Each runner serializes "
            "train, eval, then required single-GPU post-hoc diagnostics and "
            "respects gates/dependencies; generation submits nothing."
        ),
    }
    assert len(manifest["jobs"]) == 159
    gate_jobs = [job for job in manifest["jobs"] if job["kind"] == "gate_diagnostics"]
    assert len(gate_jobs) == 26
    assert all(job["gpus"] == 1 for job in gate_jobs)
    assert _run(manifest, "tinyllama", "b6", 42)["execution_policy"] == (
        "run_or_reuse_complete"
    )
    assert {
        (run["variant"], run["seed"])
        for run in manifest["runs"]
        if run["pair"] == "tinyllama" and run["variant"] == "b2_constant"
    } == {("b2_constant", 42), ("b2_constant", 43), ("b2_constant", 44)}

    conditional_runs = [run for run in manifest["runs"] if run["conditional"]]
    assert {(run["pair"], run["variant"], run["seed"]) for run in conditional_runs} == {
        (pair, variant, seed)
        for pair in ("qwen3_1p7b", "qwen25_0p5b", "llama32_1b")
        for variant in ("b1", "b2", "b3", "b5", "b6")
        for seed in (43, 44)
    }

    scheduling = manifest["scheduling"]
    assert scheduling["post_reproduction_parallel_lanes"] == 3
    hardware = {lane["id"]: lane["hardware"] for lane in scheduling["lanes"]}
    assert hardware["lane_a"]["node_profile"] == "24gx4"
    assert hardware["lane_b"]["shared_node_group"] == "bc_24gx8"
    assert hardware["lane_c"]["shared_node_group"] == "bc_24gx8"
    assert scheduling["model_availability_constraints"] == {
        "llama32_1b": "lane_a",
        "conditional_pair_lanes": {
            "llama32_1b": "lane_a",
            "qwen3_1p7b": "lane_b",
            "qwen25_0p5b": "lane_c",
        },
    }

    assert all(
        run["pipeline_lane"] == "lane_a"
        for run in manifest["runs"]
        if run["pair"] == "llama32_1b"
    )
    phase1_counts = {
        lane: sum(
            run["pipeline_lane"] == lane and not run["conditional"]
            for run in manifest["runs"]
        )
        for lane in ("lane_a", "lane_b", "lane_c")
    }
    assert phase1_counts == {"lane_a": 12, "lane_b": 13, "lane_c": 12}
    conditional_counts = {
        lane: sum(
            run["pipeline_lane"] == lane and run["conditional"]
            for run in manifest["runs"]
        )
        for lane in ("lane_a", "lane_b", "lane_c")
    }
    assert conditional_counts == {"lane_a": 10, "lane_b": 10, "lane_c": 10}
    assert {
        run["pair"]: run["pipeline_lane"]
        for run in manifest["runs"]
        if run["conditional"]
    } == {
        "llama32_1b": "lane_a",
        "qwen3_1p7b": "lane_b",
        "qwen25_0p5b": "lane_c",
    }

    all_plan_outputs: dict[str, set[str]] = {}
    run_locations: dict[str, tuple[str, str]] = {}
    plan_dependencies: dict[str, list[str]] = {}
    for lane in ("lane_a", "lane_b", "lane_c"):
        for phase in ("phase1", "conditional"):
            plan_path = output_root / "lanes" / f"{lane}.{phase}.json"
            plan = _load_json(plan_path)
            assert (plan["lane"], plan["phase"]) == (lane, phase)
            assert plan["hardware"] == hardware[lane]
            for row in plan["runs"]:
                assert "selected_checkpoint" in row["training"]
                if row["training"]["required"]:
                    provenance = row["training"]["checkpoint_provenance"]
                    assert provenance["run_id"] == row["run_id"]
                    assert provenance["git_commit"] == manifest["git_commit"]
                    assert len(provenance["train_config_sha256"]) == 64
                    assert len(provenance["split_manifest_sha256"]) == 64
                    if row["training"].get("checkpoint_directory_sha256") is None:
                        assert revision in row["training"]["selected_checkpoint"]
                assert set(row["evaluation"]["output_dirs"]) == set(suite.EVAL_LAYOUT)
                assert all(
                    revision in output
                    for output in row["evaluation"]["output_dirs"].values()
                )
                assert row["gate_diagnostics"]["batch_size"] == 1
                assert row["gate_diagnostics"]["num_samples"] == 64
                assert row["gate_diagnostics"]["required"] == (
                    row["variant"] in {"b5", "b6", "b6_constant", "b6_shuffle"}
                )
                all_plan_outputs[row["run_id"]] = set(
                    row["evaluation"]["output_dirs"].values()
                )
                run_locations[row["run_id"]] = (lane, phase)
                plan_dependencies[row["run_id"]] = row["depends_on_runs"]
    assert len(all_plan_outputs) == 67
    output_sets = list(all_plan_outputs.values())
    assert all(
        not left.intersection(right)
        for index, left in enumerate(output_sets)
        for right in output_sets[index + 1 :]
    )
    assert all(
        run_locations[dependency] == run_locations[run_id]
        for run_id, dependencies in plan_dependencies.items()
        for dependency in dependencies
    )
    assert (
        plan_dependencies[
            next(
                row["run_id"]
                for row in _load_json(output_root / "lanes" / "lane_b.phase1.json")[
                    "runs"
                ]
            )
        ]
        == []
    )

    command_rows = [
        json.loads(line)
        for line in (output_root / "recommended_commands.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(command_rows) == 6
    assert {row["lane"] for row in command_rows} == {"lane_a", "lane_b", "lane_c"}
    assert {row["phase"] for row in command_rows} == {"phase1", "conditional"}
    assert all("run-lane" in row["command"] for row in command_rows)
    assert all("--node" not in row["command"] for row in command_rows)

    stages = {stage["id"]: stage for stage in manifest["stages"]}
    assert stages["stage2_single_seed_decomposition"]["depends_on"] == [
        "stage1_reproduce_b6:gate_passed"
    ]
    assert all(
        stages[stage_id]["depends_on"] == ["stage1_reproduce_b6:gate_passed"]
        for stage_id in (
            "stage3_entropy_counterfactuals",
            "stage4_tinyllama_multiseed",
            "stage5_cross_pair_seed42",
        )
    )
    assert stages["stage5_cross_pair_multiseed"]["conditional"] is True

    analysis = _load_json(output_root / "analysis_manifest.json")
    assert len(analysis["runs"]) == 67
    questions = {row["question"] for row in analysis["component_comparisons"]}
    assert "gate_capacity" in questions
    assert "gate_capacity_static_scale_confounded" in questions
    gate = next(
        row
        for row in analysis["component_comparisons"]
        if row["question"] == "gate_capacity"
    )
    assert gate == {
        "question": "gate_capacity",
        "candidate": "b5",
        "control": "b2_constant",
    }
    reuse_example = _load_json(output_root / "reuse_overrides.step1_b6.json")
    assert reuse_example["tinyllama__b6__seed_42"]["checkpoint_dir"] == (
        "/netdisk/lijunsi/c2c-route1-identifiability/"
        "checkpoints/b6_seed42/final"
    )
    assert "evaluation_output_dirs" not in reuse_example["tinyllama__b6__seed_42"]


def test_variant_recipes_change_only_requested_controls(tmp_path: Path) -> None:
    manifest, _ = _generate(tmp_path)
    variants = (
        "b1",
        "b2",
        "b2_constant",
        "b3",
        "b4",
        "b5",
        "b6",
        "b6_constant",
        "b6_shuffle",
    )
    configs = {
        variant: _load_json(
            _run(manifest, "tinyllama", variant, 42)["training"]["config"]
        )
        for variant in variants
    }
    assert (
        len(
            {
                json.dumps(config["training"], sort_keys=True)
                for config in configs.values()
            }
        )
        == 1
    )
    assert (
        len({json.dumps(config["data"], sort_keys=True) for config in configs.values()})
        == 1
    )
    assert configs["b6"]["data"]["split_indices_path"] == (
        "recipe/train_recipe/identifiability/splits/"
        "mmlu_aux2048_seed42_april_v22.json"
    )

    b1 = configs["b1"]["model"]
    assert b1["alignment_strategy"] == "longest"
    assert "soft_alignment_top_k" not in b1
    assert b1["projector"]["params"]["alignment_confidence_gate_mode"] == "none"

    expected = {
        "b2": (1, "none", "none", "native"),
        "b2_constant": (1, "entropy", "none", "constant"),
        "b3": (4, "none", "none", "native"),
        "b4": (4, "entropy", "none", "native"),
        "b5": (1, "entropy", "token_mlp", "constant"),
        "b6": (4, "entropy", "token_mlp", "native"),
        "b6_constant": (4, "entropy", "token_mlp", "constant"),
        "b6_shuffle": (4, "entropy", "token_mlp", "shuffle"),
    }
    for variant, (top_k, confidence, gate, control) in expected.items():
        model = configs[variant]["model"]
        assert model["soft_alignment_top_k"] == top_k
        assert model["soft_alignment_confidence_mode"] == confidence
        assert model["soft_alignment_confidence_control_mode"] == control
        assert model["projector"]["params"]["alignment_confidence_gate_mode"] == gate
    assert configs["b2_constant"]["model"][
        "soft_alignment_confidence_constant_value"
    ] == pytest.approx(0.93)
    assert configs["b5"]["model"][
        "soft_alignment_confidence_constant_value"
    ] == pytest.approx(0.93)
    assert (
        configs["b6_shuffle"]["model"]["soft_alignment_confidence_shuffle_seed"] == 42
    )


def test_model_ids_final_checkpoints_and_training_runtime_command(
    tmp_path: Path,
) -> None:
    manifest, _ = _generate(tmp_path)
    expected_models = {
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "qwen3_1p7b": "Qwen/Qwen3-1.7B",
        "qwen25_0p5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "llama32_1b": "meta-llama/Llama-3.2-1B-Instruct",
    }
    for pair, model_id in expected_models.items():
        run = _run(manifest, pair, "b6", 42)
        train = _load_json(run["training"]["config"])
        assert train["model"]["base_model"] == "Qwen/Qwen3-0.6B"
        assert train["model"]["teacher_model"] == model_id
        assert run["training"]["selected_checkpoint"].endswith("/final")
        for dataset, gpu_ids in {
            "ai2-arc": [0],
            "openbookqa": [1],
            "mmlu-redux": [2, 3],
        }.items():
            config = _load_yaml(run["evaluation"]["configs"][dataset])
            assert config["eval"]["gpu_ids"] == gpu_ids
            assert config["eval"]["gate_diagnostics_mode"] == "compact"
            assert config["model"]["rosetta_config"]["teacher_model"] == model_id

    for seed, expected_name in {
        42: "mmlu_aux2048_seed42_april_v22.json",
        43: "mmlu_aux2048_seed43_seeded.json",
        44: "mmlu_aux2048_seed44_seeded.json",
    }.items():
        train = _load_json(
            _run(manifest, "tinyllama", "b6", seed)["training"]["config"]
        )
        assert train["data"]["split_indices_path"].endswith(expected_name)

    job = next(
        row for row in manifest["jobs"] if row["id"] == "train::tinyllama__b6__seed_42"
    )
    assert "torchrun" not in job["command"]
    assert job["command"][-8:-3] == [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
    ]


class _FakeProcess:
    def __init__(self, return_code: int | None):
        self.return_code = return_code
        self.terminated = False

    def poll(self) -> int | None:
        return -15 if self.terminated else self.return_code

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.poll() or 0


def test_triplet_runner_starts_all_and_fails_if_one_child_fails(tmp_path: Path) -> None:
    configs, _outputs = _make_eval_bundle(tmp_path, "triplet")
    started: list[_FakeProcess] = []

    def fake_popen(command: list[str], **_kwargs: Any) -> _FakeProcess:
        process = _FakeProcess(7 if "openbookqa.yaml" in command[-1] else None)
        started.append(process)
        return process

    assert (
        suite.run_eval_triplet(
            arc_config=Path(configs["ai2-arc"]),
            openbookqa_config=Path(configs["openbookqa"]),
            mmlu_config=Path(configs["mmlu-redux"]),
            popen_factory=fake_popen,
            sleep_fn=lambda _seconds: None,
        )
        == 1
    )
    assert len(started) == 3
    assert all(process.poll() is not None for process in started)


def test_triplet_runner_rejects_wrong_physical_gpu_assignment(tmp_path: Path) -> None:
    configs, _outputs = _make_eval_bundle(tmp_path, "wrong-gpu")
    arc_path = Path(configs["ai2-arc"])
    arc = _load_yaml(arc_path)
    arc["eval"]["gpu_ids"] = [3]
    arc_path.write_text(yaml.safe_dump(arc), encoding="utf-8")
    with pytest.raises(ValueError, match=r"expected \[0\]"):
        suite.run_eval_triplet(
            arc_config=arc_path,
            openbookqa_config=Path(configs["openbookqa"]),
            mmlu_config=Path(configs["mmlu-redux"]),
        )


def _lane_plan(tmp_path: Path) -> tuple[Path, Path, Path, list[dict[str, Any]]]:
    train_config = tmp_path / "train.json"
    train_config.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoints" / "run1" / "final"
    run1_configs, run1_outputs = _make_eval_bundle(tmp_path, "run1")
    run2_configs, run2_outputs = _make_eval_bundle(tmp_path, "run2")
    runs = [
        {
            "run_id": "run1",
            "gate_key": "reproduction",
            "depends_on_runs": [],
            "execution_policy": "run_or_reuse_complete",
            "training": {
                "required": True,
                "config": str(train_config),
                "selected_checkpoint": str(checkpoint),
            },
            "evaluation": {"configs": run1_configs, "output_dirs": run1_outputs},
        },
        {
            "run_id": "run2",
            "gate_key": "reproduction",
            "depends_on_runs": ["run1"],
            "execution_policy": "run_or_reuse_complete",
            "training": {
                "required": False,
                "config": None,
                "selected_checkpoint": None,
            },
            "evaluation": {"configs": run2_configs, "output_dirs": run2_outputs},
        },
    ]
    state = tmp_path / "state"
    plan = tmp_path / "lane.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite": "test",
                "lane": "lane_a",
                "phase": "phase1",
                "state_dir": str(state),
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    gate = tmp_path / "gates.json"
    return plan, gate, state, runs


def test_run_lane_hard_blocks_unpassed_gate(tmp_path: Path) -> None:
    plan, gate, state, _runs = _lane_plan(tmp_path)
    gate.write_text('{"reproduction":"pending"}\n', encoding="utf-8")
    calls: list[Any] = []
    assert (
        suite.run_lane_plan(
            plan_path=plan,
            gate_file=gate,
            state_dir=state,
            run_command=lambda *args, **kwargs: calls.append((args, kwargs)),
            triplet_runner=lambda **kwargs: calls.append(kwargs) or 0,
        )
        == 3
    )
    assert calls == []
    assert not (state / "FAILED.json").exists()


def test_run_lane_executes_train_then_triplet_and_marks_dependencies(
    tmp_path: Path,
) -> None:
    plan, gate, state, runs = _lane_plan(tmp_path)
    gate.write_text('{"reproduction":"pass"}\n', encoding="utf-8")
    events: list[tuple[str, str]] = []

    def fake_train(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        events.append(("train", command[0]))
        assert command[:6] == [
            "/venv/python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "script/train/SFT_train.py",
        ]
        _complete_checkpoint(Path(runs[0]["training"]["selected_checkpoint"]))
        return SimpleNamespace(returncode=0)

    def fake_triplet(**kwargs: Any) -> int:
        arc = _load_yaml(kwargs["arc_config"])
        name = Path(arc["output"]["output_dir"]).parent.name
        events.append(("eval", name))
        output_dirs = runs[0 if name == "run1" else 1]["evaluation"]["output_dirs"]
        _complete_eval_outputs(output_dirs)
        return 0

    assert (
        suite.run_lane_plan(
            plan_path=plan,
            gate_file=gate,
            state_dir=state,
            python_executable="/venv/python",
            run_command=fake_train,
            triplet_runner=fake_triplet,
            dependency_poll_seconds=0.01,
        )
        == 0
    )
    assert events == [("train", "/venv/python"), ("eval", "run1"), ("eval", "run2")]
    assert (state / "completed" / "run1.json").is_file()
    assert (state / "completed" / "run2.json").is_file()


def test_run_lane_executes_required_gate_diagnostics_inline_on_cuda0(
    tmp_path: Path,
) -> None:
    plan, gate, state, runs = _lane_plan(tmp_path)
    gate.write_text('{"reproduction":"pass"}\n', encoding="utf-8")
    run = runs[1]
    run["depends_on_runs"] = []
    artifact = tmp_path / "gate-posthoc" / "gate_diagnostics.json"
    run["gate_diagnostics"] = {
        "required": True,
        "num_samples": 64,
        "artifact": str(artifact),
        "inner_command": [
            "python",
            "script/analysis/route1_confidence_gate_diagnostics.py",
            "--batch-size",
            "1",
            "--device",
            "cuda:0",
        ],
    }
    plan_data = _load_json(plan)
    plan_data["runs"] = [run]
    plan.write_text(json.dumps(plan_data), encoding="utf-8")
    events: list[str] = []

    def evaluate(**_kwargs: Any) -> int:
        events.append("eval")
        _complete_eval_outputs(
            run["evaluation"]["output_dirs"], gate_required=True
        )
        return 0

    def run_diagnostics(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        events.append("gate")
        assert command[0] == "/venv/python"
        assert command[command.index("--batch-size") + 1] == "1"
        assert command[command.index("--device") + 1] == "cuda:0"
        _complete_posthoc_gate_diagnostics(artifact)
        return SimpleNamespace(returncode=0)

    assert (
        suite.run_lane_plan(
            plan_path=plan,
            gate_file=gate,
            state_dir=state,
            python_executable="/venv/python",
            run_command=run_diagnostics,
            triplet_runner=evaluate,
        )
        == 0
    )
    assert events == ["eval", "gate"]
    assert (state / "completed" / "run2.json").is_file()


def test_run_lane_verifies_explicit_reuse_without_launching_processes(
    tmp_path: Path,
) -> None:
    plan, gate, state, runs = _lane_plan(tmp_path)
    gate.write_text('{"reproduction":"pass"}\n', encoding="utf-8")
    reused = runs[:1]
    reused[0]["execution_policy"] = "reuse_required"
    _complete_checkpoint(Path(reused[0]["training"]["selected_checkpoint"]))
    _complete_eval_outputs(reused[0]["evaluation"]["output_dirs"])
    data = _load_json(plan)
    data["runs"] = reused
    plan.write_text(json.dumps(data), encoding="utf-8")

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("explicit reuse must not launch work")

    assert (
        suite.run_lane_plan(
            plan_path=plan,
            gate_file=gate,
            state_dir=state,
            run_command=unexpected,
            triplet_runner=unexpected,
        )
        == 0
    )
    assert (state / "completed" / "run1.json").is_file()


def test_run_lane_requires_checkpoint_reuse_but_reruns_evaluation(
    tmp_path: Path,
) -> None:
    plan, gate, state, runs = _lane_plan(tmp_path)
    gate.write_text('{"reproduction":"pass"}\n', encoding="utf-8")
    reused = runs[:1]
    reused[0]["execution_policy"] = "checkpoint_reuse_required"
    checkpoint = Path(reused[0]["training"]["selected_checkpoint"])
    _complete_checkpoint(checkpoint)
    reused[0]["training"]["checkpoint_directory_sha256"] = (
        suite._checkpoint_directory_sha256(checkpoint)
    )
    data = _load_json(plan)
    data["runs"] = reused
    plan.write_text(json.dumps(data), encoding="utf-8")
    events: list[str] = []

    def unexpected_train(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("required checkpoint reuse must skip training")

    def evaluate(**_kwargs: Any) -> int:
        events.append("eval")
        _complete_eval_outputs(reused[0]["evaluation"]["output_dirs"])
        return 0

    assert (
        suite.run_lane_plan(
            plan_path=plan,
            gate_file=gate,
            state_dir=state,
            run_command=unexpected_train,
            triplet_runner=evaluate,
        )
        == 0
    )
    assert events == ["eval"]


def test_dependency_failures_are_per_run_and_successful_retry_clears_them(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    plan = {"lane": "lane_a", "phase": "phase1"}
    failed_run = {"run_id": "failed-dependency"}
    unrelated_run = {"run_id": "unrelated"}
    suite._write_failure_marker(state, plan, failed_run, "boom")
    suite._write_failure_marker(state, plan, unrelated_run, "also boom")

    ok, reason = suite._wait_for_dependencies(
        state, ["failed-dependency"], 0.0, 0.01, lambda _seconds: None
    )
    assert not ok
    assert reason == "dependency failed: ['failed-dependency']"
    ok, reason = suite._wait_for_dependencies(
        state, [], 0.0, 0.01, lambda _seconds: None
    )
    assert ok and not reason

    suite._write_completion_marker(state, plan, failed_run)
    assert not suite._failure_marker(state, "failed-dependency").exists()
    assert suite._failure_marker(state, "unrelated").is_file()
    ok, reason = suite._wait_for_dependencies(
        state, ["failed-dependency"], 0.0, 0.01, lambda _seconds: None
    )
    assert ok and not reason


def test_run_lane_explicit_retry_overwrites_own_failure_marker(tmp_path: Path) -> None:
    plan, gate, state, runs = _lane_plan(tmp_path)
    gate.write_text('{"reproduction":"pass"}\n', encoding="utf-8")
    retry = runs[1:]
    retry[0]["depends_on_runs"] = []
    data = _load_json(plan)
    data["runs"] = retry
    plan.write_text(json.dumps(data), encoding="utf-8")
    suite._write_failure_marker(
        state, {"lane": "old-lane", "phase": "phase1"}, retry[0], "old failure"
    )

    def evaluate(**_kwargs: Any) -> int:
        _complete_eval_outputs(retry[0]["evaluation"]["output_dirs"])
        return 0

    assert (
        suite.run_lane_plan(
            plan_path=plan,
            gate_file=gate,
            state_dir=state,
            triplet_runner=evaluate,
        )
        == 0
    )
    assert not suite._failure_marker(state, "run2").exists()
    assert suite._completion_marker(state, "run2").is_file()


def test_reuse_complete_retrains_on_provenance_mismatch_and_reruns_eval(
    tmp_path: Path,
) -> None:
    plan, gate, state, runs = _lane_plan(tmp_path)
    gate.write_text('{"reproduction":"pass"}\n', encoding="utf-8")
    run = runs[0]
    expected_provenance = {
        "schema_version": 1,
        "run_id": "run1",
        "git_commit": "a" * 40,
        "train_config_sha256": "b" * 64,
        "split_manifest_sha256": "c" * 64,
        "split_indices_sha256": {"train": "d" * 64},
        "dataset_canonical_sha256": "e" * 64,
    }
    run["training"]["checkpoint_provenance"] = expected_provenance
    checkpoint = Path(run["training"]["selected_checkpoint"])
    _complete_checkpoint(checkpoint)
    suite._write_checkpoint_provenance(
        checkpoint, {**expected_provenance, "git_commit": "f" * 40}
    )
    _complete_eval_outputs(run["evaluation"]["output_dirs"])
    plan_data = _load_json(plan)
    plan_data["runs"] = [run]
    plan.write_text(json.dumps(plan_data), encoding="utf-8")
    events: list[str] = []

    def train(_command: list[str], **_kwargs: Any) -> SimpleNamespace:
        events.append("train")
        return SimpleNamespace(returncode=0)

    def evaluate(**_kwargs: Any) -> int:
        events.append("eval")
        _complete_eval_outputs(run["evaluation"]["output_dirs"])
        return 0

    assert (
        suite.run_lane_plan(
            plan_path=plan,
            gate_file=gate,
            state_dir=state,
            reuse_complete=True,
            run_command=train,
            triplet_runner=evaluate,
        )
        == 0
    )
    assert events == ["train", "eval"]
    assert suite._checkpoint_provenance_matches(checkpoint, expected_provenance)


def test_reuse_override_repoints_b6_recipe_and_lane_plan(tmp_path: Path) -> None:
    checkpoint = tmp_path / "historical" / "final"
    _complete_checkpoint(checkpoint)
    checkpoint_sha256 = suite._checkpoint_directory_sha256(checkpoint)
    _configs, outputs = _make_eval_bundle(tmp_path, "historical-b6")
    _complete_eval_outputs(outputs)
    overrides = {
        "tinyllama__b6__seed_42": {
            "mode": "reuse_required",
            "checkpoint_dir": str(checkpoint),
            "checkpoint_directory_sha256": checkpoint_sha256,
            "evaluation_output_dirs": outputs,
        }
    }
    manifest, output_root = _generate(tmp_path, overrides)
    b6 = _run(manifest, "tinyllama", "b6", 42)
    assert b6["execution_policy"] == "reuse_required"
    assert b6["training"]["selected_checkpoint"] == str(checkpoint)
    assert b6["training"]["checkpoint_directory_sha256"] == checkpoint_sha256
    plan = _load_json(output_root / "lanes" / "lane_a.phase1.json")
    entry = next(row for row in plan["runs"] if row["run_id"] == b6["id"])
    assert entry["execution_policy"] == "reuse_required"
    assert entry["training"]["checkpoint_directory_sha256"] == checkpoint_sha256
    assert entry["evaluation"]["output_dirs"] == outputs


def test_materialize_analysis_maps_receiver_seed42_to_every_seed(
    tmp_path: Path,
) -> None:
    receiver_dir = tmp_path / "receiver" / "mmlu-redux"
    b2c_dir = tmp_path / "b2c" / "mmlu-redux"
    b5_dir = tmp_path / "b5" / "mmlu-redux"
    for path in (receiver_dir, b2c_dir, b5_dir):
        path.mkdir(parents=True)
        (path / "pred_cot.csv").write_text(
            "subject,question_id,true_answer,pred,is_correct\nmain,0,A,A,true\n",
            encoding="utf-8",
        )

    analysis = {
        "schema_version": 1,
        "runs": [
            {
                "run_id": "receiver__b0__seed_42",
                "pair": "receiver",
                "variant": "b0",
                "seed": 42,
                "datasets": {
                    "mmlu-redux": {"prediction_glob": str(receiver_dir / "*_cot.csv")}
                },
            },
            {
                "run_id": "tinyllama__b2_constant__seed_43",
                "pair": "tinyllama",
                "variant": "b2_constant",
                "seed": 43,
                "datasets": {
                    "mmlu-redux": {"prediction_glob": str(b2c_dir / "*_cot.csv")}
                },
            },
            {
                "run_id": "tinyllama__b5__seed_43",
                "pair": "tinyllama",
                "variant": "b5",
                "seed": 43,
                "datasets": {
                    "mmlu-redux": {"prediction_glob": str(b5_dir / "*_cot.csv")}
                },
            },
        ],
    }
    analysis_path = tmp_path / "analysis_manifest.json"
    output_path = tmp_path / "report_manifest.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    result = suite.materialize_analysis_manifest(
        analysis_path,
        output_path,
        allow_missing=True,
    )

    assert any(
        row["method"] == "B0"
        and row["pair"] == "tinyllama"
        and row["seed"] == 43
        and row["csv"] == str((receiver_dir / "pred_cot.csv").resolve())
        for row in result["runs"]
    )
    b5 = next(row for row in result["runs"] if row["method"] == "B5")
    assert b5["receiver_csv"] == str((receiver_dir / "pred_cot.csv").resolve())
    gate = next(row for row in result["comparisons"] if row["name"] == "gate_capacity")
    assert gate == {
        "name": "gate_capacity",
        "baseline": "B2-constant",
        "candidate": "B5",
    }
    assert _load_json(output_path) == result


def test_materialize_selects_latest_complete_common_attempt_stem(
    tmp_path: Path,
) -> None:
    _configs, receiver_outputs = _make_eval_bundle(tmp_path, "receiver-complete")
    _configs, b6_outputs = _make_eval_bundle(tmp_path, "b6-complete")
    _complete_eval_outputs(receiver_outputs)
    _complete_eval_outputs(b6_outputs, gate_required=True)
    posthoc = tmp_path / "posthoc" / "gate_diagnostics.json"
    posthoc.parent.mkdir(parents=True)
    posthoc.write_text(
        json.dumps(
            {
                "status": "ok",
                "counts": {
                    "examples_seen": 64,
                    "examples_with_gate": 64,
                    "token_head_gate_projectors": 28,
                },
                "metadata": {"processed_samples": 64},
                "by_layer": {
                    str(index): {"mean": 0.5} for index in range(28)
                },
                "by_stage": {"early": {"mean": 0.5}},
                "by_layer_head": {"0/0": {"mean": 0.5}},
                "by_relative_token_bin": {"0": {"mean": 0.5}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    retry_dir = Path(b6_outputs["mmlu-redux"])
    (retry_dir / "retry_cot.csv").write_text("broken\n", encoding="utf-8")
    (retry_dir / "retry_summary.json").write_text("{}\n", encoding="utf-8")
    (retry_dir / "retry_gate_diagnostics.json").write_text(
        json.dumps({"status": "ok"}) + "\n", encoding="utf-8"
    )

    def datasets(outputs: dict[str, str]) -> dict[str, dict[str, str]]:
        return {
            task: {
                "output_dir": output_dir,
                "prediction_glob": f"{output_dir}/*_cot.csv",
                "summary_glob": f"{output_dir}/*_summary.json",
                "gate_diagnostics_glob": f"{output_dir}/*_gate_diagnostics.json",
            }
            for task, output_dir in outputs.items()
        }

    analysis = {
        "schema_version": 1,
        "runs": [
            {
                "run_id": "receiver__b0__seed_42",
                "pair": "receiver",
                "variant": "b0",
                "seed": 42,
                "datasets": datasets(receiver_outputs),
            },
            {
                "run_id": "tinyllama__b6__seed_42",
                "pair": "tinyllama",
                "variant": "b6",
                "seed": 42,
                "posthoc_gate_diagnostics": {
                    "required": True,
                    "num_samples": 64,
                    "artifact": str(posthoc),
                },
                "datasets": datasets(b6_outputs),
            },
        ],
    }
    analysis_path = tmp_path / "analysis-complete.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    result = suite.materialize_analysis_manifest(
        analysis_path, tmp_path / "materialized.json"
    )
    mmlu = next(
        row
        for row in result["runs"]
        if row["method"] == "B6" and row["task"] == "mmlu-redux"
    )
    assert Path(mmlu["csv"]).name == "run_cot.csv"
    assert Path(mmlu["gate_diagnostics"]).name == "run_gate_diagnostics.json"
    assert mmlu["gate_diagnostics_posthoc"] == str(posthoc.resolve())
