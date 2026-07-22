from __future__ import annotations

import torch

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
