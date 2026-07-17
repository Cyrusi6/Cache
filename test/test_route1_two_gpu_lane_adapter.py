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
        requested_gpus=3,
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
    assert result["hardware"]["requested_gpus"] == 3
    assert result["hardware"]["used_training_gpus"] == 2
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
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="GPU-a, 21992\nGPU-b, 8192\n"
        ),
    )
    try:
        adapter._select_startup_gpus(4096)
    except RuntimeError as error:
        assert "fewer than two" in str(error)
    else:
        raise AssertionError("busy GPU allocation was not rejected")


def test_startup_gpu_selection_uses_two_idle_cards_from_three(monkeypatch) -> None:
    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="GPU-a, 1\nGPU-b, 21992\nGPU-c, 2\n"
        ),
    )
    uuids, used = adapter._select_startup_gpus(4096)
    assert uuids == ["GPU-a", "GPU-c"]
    assert used == [1, 2]


def test_two_gpu_command_only_rewrites_distributed_training() -> None:
    training, changed = adapter._adapt_two_gpu_command(
        ["python", "-m", "torch.distributed.run", "--nproc_per_node=4", "train.py"]
    )
    assert changed
    assert "--nproc_per_node=2" in training

    diagnostics = ["python", "diagnostics.py", "--device", "cuda:0"]
    passthrough, changed = adapter._adapt_two_gpu_command(diagnostics)
    assert not changed
    assert passthrough == diagnostics
