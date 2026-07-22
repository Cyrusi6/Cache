from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import torch

from script.experiment import fpct_gpu_r2l_runner as r2l
from script.experiment.fpct_gpu_r2l_runner import exact_report, ulp_max


def test_exact_report_requires_byte_and_ulp_identity_fp32() -> None:
    left = torch.tensor([1.0, -2.0, 0.0], dtype=torch.float32)
    right = left.clone()
    report = exact_report(left, right)
    assert report["equal"] is True
    assert report["left_sha256"] == report["right_sha256"]
    assert report["max_abs"] == 0
    assert report["ulp_max"] == 0


def test_ulp_detects_bfloat16_difference() -> None:
    left = torch.tensor([1.0], dtype=torch.bfloat16)
    right = torch.nextafter(left, torch.tensor([2.0], dtype=torch.bfloat16))
    assert ulp_max(left, right) > 0
    assert exact_report(left, right)["equal"] is False


def test_actual_qwen_decode_helper_is_bitwise_on_small_cpu_case() -> None:
    row = r2l.run_actual_qwen_decode4_case(
        device=torch.device("cpu"),
        dtype=torch.float32,
        layers=2,
        kv_heads=1,
        decode_steps=1,
    )
    assert row["exact"] is True
    assert row["candidate_identity"]["candidate_kv_parent_bitwise"] is True
    assert row["semantic_parent_map_complete"] is True
    assert row["steps"][0]["logits"]["max_abs"] == 0
    assert row["steps"][0]["logits"]["ulp_max"] == 0
    assert row["steps"][0]["first_divergence"] is None


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_semantic_aggregate_requires_original_23_and_six_checks(
    tmp_path: Path, monkeypatch
) -> None:
    original = tmp_path / "original.json"
    synthetic = tmp_path / "synthetic.json"
    qwen = tmp_path / "qwen.json"
    output = tmp_path / "semantic.json"
    _write(
        original,
        {
            "status": "GO",
            "checks": {
                **{f"check_{index}": True for index in range(16)},
                "hot_path_no_sync": True,
                "latency_median": True,
                "latency_p95": True,
                "expansion_mean": True,
                "expansion_p95": True,
                "hbm": True,
                "forced_on_dk": True,
            },
            "accuracy_or_correctness_accessed": False,
        },
    )
    assert len(json.loads(original.read_text())["checks"]) == 23
    _write(
        synthetic,
        {
            "checks": {
                "native_parent_map_complete": True,
                "unknown_sidecar_fails_closed": True,
                "mixed_memory_exact_null_bitwise": True,
                "mixed_batch_exact_active_isolation": True,
            }
        },
    )
    _write(qwen, {"status": "GO", "rows": [{"exact": True}, {"exact": True}]})
    monkeypatch.setattr(
        r2l,
        "_immutable_pretrained_semantics",
        lambda *_args: {
            "exact_by_dtype": {},
            "forced_by_dtype": {},
            "precollapse_candidate_identity": {},
            "exact_ok": True,
            "active_ok": True,
            "precollapse_ok": True,
        },
    )
    payload = r2l.semantic_aggregate(
        Namespace(
            original_result=str(original),
            conditions_root=str(tmp_path),
            floors=str(tmp_path / "floors.json"),
            synthetic=str(synthetic),
            qwen_decode4=str(qwen),
            output=str(output),
        )
    )
    assert payload["status"] == "GO"
    assert all(payload["checks"].values())


def test_immutable_finalize_authorizes_training_only_after_all_gates(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.json"
    semantic = tmp_path / "semantic.json"
    active = tmp_path / "active.json"
    output = tmp_path / "final.json"
    _write(
        original,
        {
            "status": "GO",
            "checks": {f"check_{index}": True for index in range(23)},
            "accuracy_or_correctness_accessed": False,
        },
    )
    _write(
        semantic,
        {
            "status": "GO",
            "checks": {f"semantic_{index}": True for index in range(6)},
            "accuracy_or_correctness_accessed": False,
        },
    )
    _write(
        active,
        {
            "status": "DIAGNOSTIC_QUALIFIED",
            "canaries": {
                "checkpoint_native": {"qualified": True},
                "forced_on": {"qualified": True},
            },
            "accuracy_or_correctness_accessed": False,
        },
    )
    payload = r2l.immutable_finalize(
        Namespace(
            original_result=str(original),
            semantic_result=str(semantic),
            active_aggregate=str(active),
            output=str(output),
        )
    )
    assert payload["status"] == "GO"
    assert payload["classification"] == "R2L_IMMUTABLE_GO"
    assert payload["training_authorized"] is True
