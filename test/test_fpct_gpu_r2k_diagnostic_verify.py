from __future__ import annotations

import math

import pytest

from script.analysis.fpct_gpu_r2k_diagnostic_verify import (
    bootstrap_ucb,
    parse_gpu,
    percentile,
)


def test_percentile_interpolates_and_rejects_empty() -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.5
    assert percentile([1.0, 3.0], 0.95) == pytest.approx(2.9)
    with pytest.raises(ValueError, match="empty percentile"):
        percentile([], 0.95)


def test_bootstrap_ucb_is_deterministic_and_bounded() -> None:
    values = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03]
    first = bootstrap_ucb(values)
    second = bootstrap_ucb(values)
    assert first == second
    assert min(values) <= first <= max(values)


def test_parse_gpu_requires_complete_finite_telemetry() -> None:
    parsed = parse_gpu(
        "GPU-abc, NVIDIA GeForce RTX 4090, P0, 2745, 10501, 53, 217.73, 450, 0x0000000000000000"
    )
    assert parsed["uuid"] == "GPU-abc"
    assert parsed["pstate"] == "P0"
    assert math.isclose(parsed["power_w"], 217.73)
    with pytest.raises(ValueError, match="not parseable"):
        parse_gpu("unavailable")
