from __future__ import annotations

import json
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import script.analysis.route1_phase15_interventions as phase15
from rosetta.model.projector import C2CProjector
from rosetta.utils.eval_interventions import (
    apply_eval_intervention_to_config,
    apply_projector_eval_intervention,
    build_eval_intervention_provenance,
    normalize_eval_intervention,
)
from script.analysis.route1_phase15_interventions import (
    ANOMALY_INTERVENTIONS,
    DATASETS,
    INTERVENTIONS,
    PAIRS,
    SEEDS,
    generate_manifest,
    generate_qwen25_seed44_anomaly_manifest,
)


def _eval_config() -> dict:
    return {
        "model": {
            "model_name": "Rosetta",
            "rosetta_config": {
                "base_model": "receiver",
                "teacher_model": "sender",
                "soft_alignment_top_k": 4,
                "soft_alignment_confidence_control_mode": "native",
                "checkpoints_dir": "checkpoint",
            },
        },
        "output": {"output_dir": "out"},
        "eval": {"dataset": "mmlu-redux", "gpu_ids": [0]},
    }


def test_eval_config_intervention_materializes_only_eval_fields() -> None:
    config = _eval_config()
    normalized = apply_eval_intervention_to_config(
        config,
        {
            "id": "counterfactual",
            "top_k": 1,
            "entropy_mode": "constant",
            "entropy_constant_value": 0.93,
            "gate_mode": "static",
        },
    )

    assert normalized == {
        "schema_version": 1,
        "id": "counterfactual",
        "scope": "evaluation_only",
        "top_k": 1,
        "entropy_mode": "constant",
        "gate_mode": "static",
        "gate_components": {
            "alignment_confidence": "static",
            "legacy_scalar_kv": "checkpoint_native",
        },
        "entropy_constant_value": 0.93,
    }
    rosetta = config["model"]["rosetta_config"]
    assert rosetta["soft_alignment_top_k"] == 1
    assert rosetta["soft_alignment_confidence_control_mode"] == "constant"
    assert rosetta["soft_alignment_confidence_constant_value"] == 0.93
    assert config["eval"]["intervention"] == normalized
    once = copy.deepcopy(config)
    assert apply_eval_intervention_to_config(config) == normalized
    assert config == once


def test_eval_intervention_validation_is_strict() -> None:
    with pytest.raises(ValueError, match="entropy_constant_value is required"):
        normalize_eval_intervention({"entropy_mode": "constant"})
    with pytest.raises(ValueError, match="top_k must be one of"):
        normalize_eval_intervention({"top_k": 2})
    with pytest.raises(ValueError, match="entropy_shuffle_seed is required"):
        normalize_eval_intervention({"entropy_mode": "shuffled"})


def _projector() -> C2CProjector:
    return C2CProjector(
        source_dim=4,
        target_dim=4,
        source_num_heads=1,
        target_num_heads=1,
        intermediate_dim=4,
        hidden_dim=4,
        num_layers=3,
        dropout=0.0,
        alignment_confidence_gate_mode="token_mlp",
    )


def _compute_gate(projector: C2CProjector) -> tuple[torch.Tensor, torch.Tensor]:
    return projector._compute_alignment_confidence(
        source_confidence=torch.tensor([[0.25, 0.75]]),
        source_weights=torch.tensor([[[1.0, 0.0], [0.5, 0.5]]]),
        source_entropy=torch.tensor([[0.0, 1.0]]),
        source_entropy_override=torch.tensor([[True, True]]),
        key_hidden=torch.zeros(1, 2, 4),
        value_hidden=torch.zeros(1, 2, 4),
        target_shape=(1, 1, 2, 4),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_same_checkpoint_gate_views_static_and_forced_on() -> None:
    projector = _projector()
    projector.eval()
    model = SimpleNamespace(projector_list=[projector])

    result = apply_projector_eval_intervention(
        model, {"id": "static", "gate_mode": "static"}
    )
    static_key, static_value = _compute_gate(projector)
    assert result == {
        "gate_mode": "static",
        "gate_components": {
            "alignment_confidence": "static",
            "legacy_scalar_kv": "checkpoint_native",
        },
        "projector_count": 1,
    }
    assert static_key.flatten().tolist() == pytest.approx([0.25, 0.75])
    assert torch.equal(static_key, static_value)
    projector.eval()
    with torch.no_grad():
        projector.key_gate_logit.fill_(-1.0)
        projector.value_gate_logit.fill_(-1.0)
        source = torch.zeros(1, 1, 2, 4)
        target = torch.zeros_like(source)
        projector(
            (source, source),
            (target, target),
            source_confidence=torch.tensor([[0.25, 0.75]]),
            source_weights=torch.tensor([[[1.0, 0.0], [0.5, 0.5]]]),
        )
    assert projector.last_legacy_key_gate.item() == pytest.approx(0.0)
    assert projector.last_legacy_value_gate.item() == pytest.approx(0.0)

    apply_projector_eval_intervention(
        model, {"id": "forced", "gate_mode": "forced_on"}
    )
    forced_key, forced_value = _compute_gate(projector)
    assert torch.equal(forced_key, torch.ones_like(forced_key))
    assert torch.equal(forced_key, forced_value)
    with torch.no_grad():
        projector(
            (source, source),
            (target, target),
            source_confidence=torch.tensor([[0.25, 0.75]]),
            source_weights=torch.tensor([[[1.0, 0.0], [0.5, 0.5]]]),
        )
    assert projector.last_legacy_key_gate.item() == pytest.approx(1.0)
    assert projector.last_legacy_value_gate.item() == pytest.approx(1.0)


def test_optional_gate_isolation_modes_do_not_change_main_forced_on_semantics() -> None:
    projector = _projector()
    projector.eval()
    model = SimpleNamespace(projector_list=[projector])
    with torch.no_grad():
        projector.key_gate_logit.fill_(-1.0)
        projector.value_gate_logit.fill_(-1.0)
    source = torch.zeros(1, 1, 2, 4)
    target = torch.zeros_like(source)
    confidence = torch.tensor([[0.25, 0.75]])
    weights = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])

    apply_projector_eval_intervention(
        model, {"id": "align", "gate_mode": "alignment_forced_on"}
    )
    alignment_gate, _ = _compute_gate(projector)
    with torch.no_grad():
        projector(
            (source, source),
            (target, target),
            source_confidence=confidence,
            source_weights=weights,
        )
    assert torch.equal(alignment_gate, torch.ones_like(alignment_gate))
    assert projector.last_legacy_key_gate.item() == pytest.approx(0.0)

    apply_projector_eval_intervention(
        model, {"id": "legacy", "gate_mode": "legacy_forced_on"}
    )
    learned_gate, _ = _compute_gate(projector)
    with torch.no_grad():
        projector(
            (source, source),
            (target, target),
            source_confidence=confidence,
            source_weights=weights,
        )
    assert learned_gate.flatten().tolist() == pytest.approx([0.25, 0.75])
    assert projector.last_legacy_key_gate.item() == pytest.approx(1.0)


def test_intervention_provenance_records_checkpoint_without_training_mutation(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("eval: {}\n", encoding="utf-8")
    config = _eval_config()
    config["eval"]["intervention"] = {
        "id": "b3_eval_k1",
        "top_k": 1,
    }
    apply_eval_intervention_to_config(config)

    provenance = build_eval_intervention_provenance(
        config=config, config_path=config_path, argv=["python", "evaluator.py"]
    )

    assert provenance is not None
    assert provenance["checkpoint_dir"] == "checkpoint"
    assert provenance["training_state_mutated"] is False
    assert len(provenance["config_sha256"]) == 64
    assert len(provenance["provenance_sha256"]) == 64


def _write_phase1_fixture(root: Path) -> tuple[Path, Path]:
    manifest_runs = []
    analysis_runs = []
    for pair in PAIRS:
        for seed in SEEDS:
            for variant in ("b2", "b3", "b6"):
                run_id = f"{pair}__{variant}__seed_{seed}"
                checkpoint = root / "checkpoints" / run_id / "final"
                checkpoint.mkdir(parents=True)
                (checkpoint / "projector_0.pt").write_bytes(b"weights")
                configs = {}
                datasets = {}
                for dataset in DATASETS:
                    config_path = root / "phase1_eval" / run_id / f"{dataset}.yaml"
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    config = _eval_config()
                    config["model"]["rosetta_config"]["checkpoints_dir"] = str(
                        checkpoint
                    )
                    config["output"]["output_dir"] = str(
                        root / "phase1_results" / run_id / dataset
                    )
                    config["eval"]["dataset"] = dataset
                    config["eval"]["gpu_ids"] = [0]
                    config_path.write_text(
                        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
                    )
                    configs[dataset] = str(config_path.relative_to(root))
                    prediction = (
                        root / "phase1_results" / run_id / dataset / "native_cot.csv"
                    )
                    prediction.parent.mkdir(parents=True, exist_ok=True)
                    prediction.write_text("question_id,pred\n1,A\n", encoding="utf-8")
                    datasets[dataset] = {
                        "prediction_glob": str(
                            prediction.relative_to(root).parent / "*_cot.csv"
                        )
                    }
                manifest_runs.append(
                    {
                        "id": run_id,
                        "pair": pair,
                        "variant": variant,
                        "seed": seed,
                        "training": {
                            "selected_checkpoint": str(checkpoint.relative_to(root)),
                            "checkpoint_directory_sha256": "a" * 64,
                        },
                        "evaluation": {"configs": configs},
                    }
                )
                analysis_runs.append(
                    {
                        "run_id": run_id,
                        "pair": pair,
                        "variant": variant,
                        "seed": seed,
                        "datasets": datasets,
                    }
                )
    manifest_path = root / "phase1_manifest.json"
    analysis_path = root / "phase1_analysis_manifest.json"
    manifest_path.write_text(
        json.dumps({"runs": manifest_runs}), encoding="utf-8"
    )
    analysis_path.write_text(
        json.dumps({"runs": analysis_runs}), encoding="utf-8"
    )
    return manifest_path, analysis_path


def test_phase15_manifest_generates_only_72_non_native_triplets(
    tmp_path: Path,
) -> None:
    phase1_manifest, phase1_analysis = _write_phase1_fixture(tmp_path)
    output_root = tmp_path / "generated"
    manifest = generate_manifest(
        phase1_manifest_path=phase1_manifest,
        phase1_analysis_manifest_path=phase1_analysis,
        phase1_artifact_root=tmp_path,
        output_root=output_root,
        results_root=tmp_path / "results/phase15",
        recommended_shards=7,
    )

    assert manifest["summary"]["new_triplet_count"] == 72
    assert manifest["summary"]["new_dataset_eval_count"] == 216
    assert len(manifest["runs"]) == len(PAIRS) * len(SEEDS) * len(INTERVENTIONS)
    assert sum(
        shard["run_count"] for shard in manifest["scheduling"]["commands"]
    ) == 72
    assert [
        shard["run_count"] for shard in manifest["scheduling"]["commands"]
    ] == [11, 11, 10, 10, 10, 10, 10]
    assert all(
        shard["gpus"] == 2 for shard in manifest["scheduling"]["commands"]
    )
    assert all(run["checkpoint"]["same_checkpoint_no_training"] for run in manifest["runs"])
    assert all("train" not in run for run in manifest["runs"])
    for run in manifest["runs"]:
        for output_dir in run["output_dirs"].values():
            output_path = Path(output_dir)
            assert output_path.is_absolute()
            assert output_path.is_dir()

    b2_k4 = next(run for run in manifest["runs"] if run["intervention"]["id"] == "b2_eval_k4")
    assert b2_k4["intervention"]["top_k"] == 4
    assert b2_k4["native_comparator"]["variant"] == "b2"
    assert b2_k4["ambiguity_source"]["variant"] == "b3"
    assert all(Path(path).is_file() for path in b2_k4["native_comparator"]["prediction_csv"].values())

    static = next(run for run in manifest["runs"] if run["intervention"]["id"] == "b6_gate_static")
    assert static["intervention"]["gate_mode"] == "static"
    assert static["ambiguity_source"]["variant"] == "b6"
    assert set(static["output_dirs"].values()).isdisjoint(
        static["native_comparator"]["prediction_csv"].values()
    )


def test_two_gpu_triplet_runs_small_tasks_then_mmlu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = {}
    for dataset, gpu_ids in phase15.GPU_LAYOUT.items():
        path = tmp_path / f"{dataset}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "eval": {"dataset": dataset, "gpu_ids": gpu_ids},
                    "output": {"output_dir": str(tmp_path / "outputs" / dataset)},
                }
            ),
            encoding="utf-8",
        )
        configs[dataset] = path

    events = []

    class Process:
        def __init__(self, command, cwd):
            events.append(("popen", Path(command[-1]).name, cwd))

        def poll(self):
            return 0

        def terminate(self):
            events.append(("terminate",))

        def wait(self):
            return 0

    monkeypatch.setattr(phase15.subprocess, "Popen", Process)

    def run(command, cwd, check):
        events.append(("run", Path(command[-1]).name, cwd, check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(phase15.subprocess, "run", run)

    return_code = phase15.run_triplet(
        arc_config=configs["ai2-arc"],
        openbookqa_config=configs["openbookqa"],
        mmlu_config=configs["mmlu-redux"],
    )

    assert return_code == 0
    assert [event[:2] for event in events] == [
        ("popen", "ai2-arc.yaml"),
        ("popen", "openbookqa.yaml"),
        ("run", "mmlu-redux.yaml"),
    ]


def test_two_gpu_triplet_resumes_completed_datasets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = {}
    for dataset, gpu_ids in (
        ("ai2-arc", [0]),
        ("openbookqa", [1]),
        ("mmlu-redux", [0, 1]),
    ):
        config = _eval_config()
        output_dir = tmp_path / "outputs" / dataset
        config["eval"]["dataset"] = dataset
        config["eval"]["gpu_ids"] = gpu_ids
        config["output"]["output_dir"] = str(output_dir)
        path = tmp_path / f"{dataset}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        configs[dataset] = path
        if dataset != "openbookqa":
            output_dir.mkdir(parents=True)
            (output_dir / "run_cot.csv").write_text("x\n", encoding="utf-8")
            (output_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")
            (output_dir / "eval_intervention_provenance.json").write_text(
                "{}\n", encoding="utf-8"
            )

    events = []

    class Process:
        def __init__(self, command, cwd):
            events.append(("popen", Path(command[-1]).name, cwd))

        def poll(self):
            return 0

        def terminate(self):
            events.append(("terminate",))

        def wait(self):
            return 0

    monkeypatch.setattr(phase15.subprocess, "Popen", Process)

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("completed MMLU must not be re-run")

    monkeypatch.setattr(phase15.subprocess, "run", unexpected_run)

    assert phase15.run_triplet(
        arc_config=configs["ai2-arc"],
        openbookqa_config=configs["openbookqa"],
        mmlu_config=configs["mmlu-redux"],
    ) == 0
    assert [event[:2] for event in events] == [
        ("popen", "openbookqa.yaml")
    ]


def test_tracked_phase15_recipe_fixes_72_runs_and_three_node_pool_jobs() -> None:
    recipe_path = (
        Path(__file__).resolve().parents[1]
        / "recipe/eval_recipe/phase1_5/route1_phase15_interventions.json"
    )
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))

    assert recipe["source_commit"].startswith("0d30852")
    assert recipe["expansion"]["checkpoint_selection_record_count"] == 12
    assert recipe["expansion"]["new_triplet_count"] == 72
    assert len(recipe["runs"]) == 72
    assert recipe["scheduling"]["shard_count"] == 7
    assert recipe["scheduling"]["gpus_per_shard"] == 2
    assert recipe["scheduling"]["physical_job_count"] == 3
    assert recipe["scheduling"]["node_allocation"] == {
        "4090-24gx4": [0, 1],
        "4090-24gx8": [2, 3, 4, 5],
        "4090-48gx2": [6],
    }
    assert recipe["scheduling"]["run_counts"] == [11, 11, 10, 10, 10, 10, 10]


def test_optional_qwen25_seed44_anomaly_manifest_has_only_two_triplets(
    tmp_path: Path,
) -> None:
    phase1_manifest, phase1_analysis = _write_phase1_fixture(tmp_path)
    manifest = generate_qwen25_seed44_anomaly_manifest(
        phase1_manifest_path=phase1_manifest,
        phase1_analysis_manifest_path=phase1_analysis,
        phase1_artifact_root=tmp_path,
        output_root=tmp_path / "anomaly",
        results_root=tmp_path / "results/phase15_anomaly",
    )

    assert manifest["summary"]["new_triplet_count"] == 2
    assert manifest["summary"]["new_dataset_eval_count"] == 6
    assert len(manifest["runs"]) == len(ANOMALY_INTERVENTIONS) == 2
    assert {run["intervention"]["gate_mode"] for run in manifest["runs"]} == {
        "alignment_forced_on",
        "legacy_forced_on",
    }
    assert manifest["scheduling"]["recommended_shards"] == 1
    assert manifest["scheduling"]["gpu_per_shard"] == 2

    tracked = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "recipe/eval_recipe/phase1_5/qwen25_seed44_gate_anomaly.json"
        ).read_text(encoding="utf-8")
    )
    assert tracked["main_72_matrix_excluded"] is True
    assert len(tracked["runs"]) == 2
