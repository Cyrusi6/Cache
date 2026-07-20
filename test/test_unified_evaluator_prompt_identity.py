from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

from script.evaluation import unified_evaluator as evaluator_module


def _config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"model_name": "Qwen/Qwen3-0.6B"},
                "eval": {
                    "dataset": "ai2-arc",
                    "gpu_ids": [0],
                    "answer_method": "generate",
                    "use_cot": False,
                    "use_template": True,
                },
                "output": {"output_dir": str(path.parent / "output")},
            }
        ),
        encoding="utf-8",
    )


def test_main_verifies_prompt_identity_before_cuda_visibility_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "eval.yaml"
    _config(config)
    events: list[str] = []

    class FakeEvaluator:
        def __init__(self, _config_value):
            events.append("init")

        def preflight_prompt_identity(self):
            events.append("preflight")

        def run(self):
            events.append("run")

    monkeypatch.setattr(evaluator_module, "UnifiedEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        evaluator_module,
        "configure_cuda_visibility",
        lambda: events.append("cuda_visibility") or False,
    )
    monkeypatch.setattr(sys, "argv", ["unified_evaluator.py", "--config", str(config)])

    evaluator_module.main()

    assert events == ["init", "preflight", "cuda_visibility", "run"]


def test_freeze_manifest_exits_before_cuda_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "eval.yaml"
    output = tmp_path / "prompt_identity.json"
    _config(config)
    events: list[str] = []

    class FakeEvaluator:
        def __init__(self, _config_value):
            events.append("init")

        def freeze_prompt_identity_manifest(self, path):
            assert Path(path) == output
            events.append("freeze")

        def preflight_prompt_identity(self):
            events.append("preflight")

        def run(self):
            events.append("run")

    monkeypatch.setattr(evaluator_module, "UnifiedEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        evaluator_module,
        "configure_cuda_visibility",
        lambda: events.append("cuda_visibility") or False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unified_evaluator.py",
            "--config",
            str(config),
            "--freeze-prompt-identity-manifest",
            str(output),
        ],
    )

    evaluator_module.main()

    assert events == ["init", "freeze"]
