from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from script.k8s import route1_two_gpu_lane_adapter as adapter


def test_materialize_two_gpu_plan_preserves_effective_global_batch(
    tmp_path: Path, monkeypatch
) -> None:
    source_prefix = Path("local/tmp/source-suite")
    adapted_prefix = Path("local/tmp/adapted-suite")
    monkeypatch.setattr(adapter, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(adapter, "SOURCE_PREFIX", source_prefix)
    monkeypatch.setattr(adapter, "ADAPTED_PREFIX", adapted_prefix)

    def checkpoint_provenance_contract(**kwargs):
        config_path = Path(kwargs["train_config_path"])
        return {
            "schema_version": 1,
            "run_id": kwargs["run_id"],
            "git_commit": kwargs["git_commit"],
            "train_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        }

    fake_suite = SimpleNamespace(
        _git_commit_sha=lambda _root: "a" * 40,
        _checkpoint_provenance_contract=checkpoint_provenance_contract,
    )
    monkeypatch.setattr(adapter, "_load_suite", lambda: fake_suite)

    source_root = tmp_path / source_prefix
    train_path = source_root / "train/run.json"
    train_path.parent.mkdir(parents=True)
    train_path.write_text(
        json.dumps(
            {
                "training": {
                    "num_processes": 4,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 8,
                }
            }
        ),
        encoding="utf-8",
    )

    configs = {}
    for dataset, gpu_ids in {
        "ai2-arc": [0],
        "openbookqa": [1],
        "mmlu-redux": [2, 3],
    }.items():
        path = source_root / f"eval/run/{dataset}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump({"eval": {"dataset": dataset, "gpu_ids": gpu_ids}}),
            encoding="utf-8",
        )
        configs[dataset] = path.relative_to(tmp_path).as_posix()

    plan_path = source_root / "lanes/lane_b.phase1.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        json.dumps(
            {
                "lane": "lane_b",
                "phase": "phase1",
                "state_dir": "local/tmp/state",
                "runs": [
                    {
                        "run_id": "run",
                        "training": {
                            "required": True,
                            "config": train_path.relative_to(tmp_path).as_posix(),
                            "selected_checkpoint": "local/checkpoints/run/final",
                        },
                        "evaluation": {
                            "configs": configs,
                            "output_dirs": {
                                dataset: f"local/results/run/{dataset}"
                                for dataset in configs
                            },
                        },
                        "gate_diagnostics": {
                            "output_dir": "local/results/run/gate",
                            "inner_command": [
                                "python",
                                "diag.py",
                                "--eval-config",
                                configs["mmlu-redux"],
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    adapted_plan_path = tmp_path / adapted_prefix / "lanes/lane_b.phase1.json"

    result = adapter.materialize(
        plan_path,
        adapted_plan_path,
        node_profile="24gx2",
        gpu_memory_gib=24,
    )

    run = result["runs"][0]
    adapted_train_path = tmp_path / run["training"]["config"]
    adapted_train = json.loads(adapted_train_path.read_text(encoding="utf-8"))
    training = adapted_train["training"]
    assert training["num_processes"] == 2
    assert training["per_device_train_batch_size"] == 1
    assert training["gradient_accumulation_steps"] == 16
    assert (
        training["num_processes"]
        * training["per_device_train_batch_size"]
        * training["gradient_accumulation_steps"]
        == 32
    )
    assert run["training"]["checkpoint_provenance"]["train_config_sha256"] == (
        hashlib.sha256(adapted_train_path.read_bytes()).hexdigest()
    )
    assert result["hardware"]["requested_gpus"] == 2
    assert result["hardware"]["effective_global_batch_size"] == 32

    for dataset, expected in adapter.GPU_LAYOUT.items():
        config = yaml.safe_load(
            (tmp_path / run["evaluation"]["configs"][dataset]).read_text(
                encoding="utf-8"
            )
        )
        assert config["eval"]["gpu_ids"] == expected
    assert run["gate_diagnostics"]["inner_command"][-1] == (
        run["evaluation"]["configs"]["mmlu-redux"]
    )


def test_startup_gpu_memory_check_rejects_busy_allocation(monkeypatch) -> None:
    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="1\n21992\n"),
    )
    try:
        adapter._check_startup_gpu_memory(4096)
    except RuntimeError as error:
        assert "used_mib=[1, 21992]" in str(error)
    else:
        raise AssertionError("busy GPU allocation was not rejected")
