from __future__ import annotations

from types import SimpleNamespace

import pytest

from script.evaluation.unified_evaluator import (
    configure_cuda_visibility,
    summarize_alignment_diagnostics,
    validate_worker_completion,
)


def test_summarize_alignment_diagnostics_reduces_message_tokens() -> None:
    details = {
        "message_mask": [False, True, True, True],
        "soft_alignment": {
            "source_indices": [[-1, -1], [3, -1], [4, 5], [6, 7]],
            "source_weights": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.75, 0.25]],
            "source_confidence": [1.0, 1.0, 0.5, 0.75],
            "source_entropy": [0.0, 0.0, 1.0, 0.8112781244591328],
            "fallback_mask": [False, False, False, True],
            "top1_boundary_hit_mask": [False, True, False, True],
        },
    }

    summary = summarize_alignment_diagnostics(details)

    assert summary["alignment_bucket"] == "one-to-many"
    assert summary["candidate_count"] == pytest.approx(5 / 3)
    assert summary["candidate_count_max"] == 2
    assert summary["one_to_many_rate"] == pytest.approx(2 / 3)
    assert summary["alignment_entropy"] == pytest.approx(
        (0.0 + 1.0 + 0.8112781244591328) / 3
    )
    assert summary["boundary_mismatch"] == pytest.approx(1 / 3)
    assert summary["confidence"] == pytest.approx(0.75)
    assert summary["fallback_rate"] == pytest.approx(1 / 3)


def test_validate_worker_completion_rejects_partial_results() -> None:
    processes = [SimpleNamespace(exitcode=0), SimpleNamespace(exitcode=1)]
    with pytest.raises(RuntimeError, match="failed=.*missing_results"):
        validate_worker_completion(processes, {0: {"all_cors": []}})


def test_validate_worker_completion_accepts_all_ranks() -> None:
    processes = [SimpleNamespace(exitcode=0), SimpleNamespace(exitcode=0)]
    validate_worker_completion(processes, {0: {}, 1: {}})


def test_configure_cuda_visibility_preserves_explicit_scheduler_uuid_mask() -> None:
    env = {
        "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
        "C2C_PRESERVE_CUDA_VISIBLE_DEVICES": "1",
    }

    assert configure_cuda_visibility(env) is True
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"


def test_configure_cuda_visibility_keeps_legacy_default() -> None:
    env = {"CUDA_VISIBLE_DEVICES": "0,1"}

    assert configure_cuda_visibility(env) is False
    assert "CUDA_VISIBLE_DEVICES" not in env
