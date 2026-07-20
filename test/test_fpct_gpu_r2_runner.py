from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from script.experiment import fpct_gpu_r2_runner as runner


ROOT = Path(__file__).resolve().parents[1]


def test_operator_and_profile_panels_are_frozen() -> None:
    assert list(runner.CONDITIONS) == [
        "OP01_CPOST_NATIVE", "OP02_F_NATIVE", "OP03_F_REP_NATIVE",
        "OP04_F_FORCED", "OP05_F_REP_FORCED", "OP06_F_BYPASS",
        "OP07_M1_CPOST", "OP08_M1_F",
    ]
    assert list(runner.PROFILE_CONDITIONS) == [
        "P2_CPOST_OFF", "P3_F_OFF", "P4_F_REPLICATED",
        "P5_F_ON", "P6_DECODE4",
    ]
    assert runner.TRACE_DTYPES == {
        "fp32": runner.torch.float32,
        "bf16": runner.torch.bfloat16,
    }


def test_condition_config_is_explicit_eager_and_seeded() -> None:
    lock = runner.load_json(
        ROOT / "recipe/eval_recipe/fpct_confirmatory/confirmatory_run_lock.json"
    )
    config = runner._model_config(lock, runner.CONDITIONS["OP05_F_REP_FORCED"])
    assert config["attn_implementation"] == "eager"
    assert config["projector_init_seed"] == 104729
    assert config["fpct_operator"] == "f"
    assert config["fpct_replicated_collapse"] is True
    assert config["fpct_instrumentation"] is True
    assert config["fpct_trace"] is True


def test_runner_never_mutates_operator_after_forward() -> None:
    source = inspect.getsource(runner)
    assert ".fpct_operator =" not in source
    assert ".fpct_replicated_collapse =" not in source
    assert "operator-condition" in inspect.getsource(runner.pretrained_matrix)
    assert "_sealed_subprocess" in inspect.getsource(runner.pretrained_matrix)


def test_metric_trace_reader_preserves_layer_identity() -> None:
    trace = {
        "rows": [{
            "trace": {"layers": {
                "0": {"d_k": 0.25, "mechanism": {"jensen_gap/max": 0.5}},
                "27": {"d_k": 0.75, "mechanism": {"jensen_gap/max": 1.0}},
            }}
        }]
    }
    assert runner._metric_from_trace(trace, "d_k") == [(0, 0.25), (27, 0.75)]
    assert runner._metric_from_trace(trace, "jensen_gap") == [(0, 0.5), (27, 1.0)]


def test_compact_trace_identity_and_first_divergence() -> None:
    def rows(delta: float):
        return [{
            "panel_id": "p0",
            "layers": {
                "0": {
                    "fused_candidate_key": runner.torch.tensor([1.0]),
                    "fused_candidate_value": runner.torch.tensor([2.0]),
                    "pre_o_proj_last": runner.torch.tensor([0.0]),
                    "post_o_proj_last": runner.torch.tensor([0.0]),
                    "residual_last": runner.torch.tensor([0.0]),
                },
                "1": {
                    "fused_candidate_key": runner.torch.tensor([1.0]),
                    "fused_candidate_value": runner.torch.tensor([2.0]),
                    "pre_o_proj_last": runner.torch.tensor([delta]),
                    "post_o_proj_last": runner.torch.tensor([delta]),
                    "residual_last": runner.torch.tensor([delta]),
                },
            },
        }]

    assert runner._precollapse_identity(rows(0.0), rows(1.0))
    report = runner._layer_delta_report(rows(0.0), rows(0.1), tolerance=0.05)
    assert report["first_above_tolerance"]["residual_last"]["layer"] == 1
    assert report["max_abs"]["post_o_proj_last"] == pytest.approx(0.1)
