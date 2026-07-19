from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "script/analysis/fpct_1b_structural_support_audit.py"
SPEC = importlib.util.spec_from_file_location("fpct_1b_structural_support_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_frozen_content_hash_matches_phase2a_canonical_encoding() -> None:
    digest = audit.content_hash("  Question   text ", [" A ", "B", "C", "D", "ignored"])
    assert digest == audit.content_hash("Question text", ["A", "B", "C", "D"])
    assert len(digest) == 64


def test_split_assignment_is_group_deterministic() -> None:
    group = audit.content_hash("q", ["a", "b", "c", "d"])
    first = audit.split_for_content(group)
    assert first in {"fit", "calibration", "model-selection", "test"}
    assert audit.split_for_content(group) == first


def test_zero_support_contract() -> None:
    row, failure = audit.classify_candidate_row(
        [-1, -1, -1, -1], [0.0, 0.0, 0.0, 0.0],
        source_length=3, source_padding_mask=[False, False, False],
        fallback_flag=False,
    )
    assert not failure
    assert row["legal_candidate_count"] == 0
    assert row["candidate_count_stratum"] == "m0_zero_support"
    assert row["a_max"] is None
    assert row["secondary_mass"] is None
    assert row["n_eff"] is None
    assert row["is_primary_structural_m2"] == 0


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_uniform_candidate_strata(m: int) -> None:
    indices = list(range(m)) + [-1] * (4 - m)
    weights = [1.0 / m] * m + [0.0] * (4 - m)
    row, failure = audit.classify_candidate_row(
        indices, weights, source_length=4,
        source_padding_mask=[False] * 4, fallback_flag=(m == 1),
    )
    assert not failure
    assert row["legal_candidate_count"] == m
    assert row["is_primary_structural_m2"] == int(m >= 2)
    assert row["is_high_cardinality_m3"] == int(m >= 3)
    assert row["is_strict_m4"] == int(m == 4)
    assert row["a_max"] == pytest.approx(1.0 / m)
    assert row["secondary_mass"] == pytest.approx(1.0 - 1.0 / m)
    assert row["n_eff"] == pytest.approx(float(m))


def test_duplicate_legal_source_index_is_integrity_failure() -> None:
    row, failure = audit.classify_candidate_row(
        [1, 1, -1, -1], [0.5, 0.5, 0.0, 0.0],
        source_length=3, source_padding_mask=[False] * 3, fallback_flag=False,
    )
    assert failure
    assert row["duplicate_source_index_count"] == 1


def test_invalid_positive_nonfinite_and_negative_mass_are_failures() -> None:
    invalid, invalid_failure = audit.classify_candidate_row(
        [5, -1, -1, -1], [1.0, 0.0, 0.0, 0.0],
        source_length=2, source_padding_mask=[False, False], fallback_flag=False,
    )
    assert invalid_failure
    assert invalid["invalid_positive_mass_count"] == 1

    bad, bad_failure = audit.classify_candidate_row(
        [0, 1, -1, -1], [float("nan"), -0.1, 0.0, 0.0],
        source_length=2, source_padding_mask=[False, False], fallback_flag=False,
    )
    assert bad_failure
    assert bad["nonfinite_weight_count"] == 1
    assert bad["negative_weight_count"] == 1


def _group(pair: str, task: str, positive: int, total: int) -> list[dict[str, str]]:
    return [
        {
            "pair": pair,
            "task": task,
            "split": "fit" if index % 2 == 0 else "calibration",
            "has_primary_structural_m2": str(int(index < positive)),
            "has_high_cardinality_m3": "0",
            "has_strict_m4": "0",
            "integrity_failure": "0",
        }
        for index in range(total)
    ]


def test_readiness_and_ranking_exclude_same_tokenizer_control() -> None:
    rows: list[dict[str, str]] = []
    positives = {
        "tinyllama": (35, 35, 35),
        "qwen25_0p5b": (40, 30, 40),
        "llama32_1b": (5, 5, 5),
        "qwen3_1p7b": (90, 90, 90),
    }
    for pair, task_counts in positives.items():
        for task, count in zip(audit.TASK_ORDER, task_counts):
            rows.extend(_group(pair, task, count, 100))
    readiness, status, ranking, selected = audit.derive_readiness(rows)
    assert status == "CROSS_PAIR_PILOT_READY"
    assert ranking == ["tinyllama", "qwen25_0p5b"]
    assert selected == "tinyllama"
    control = next(row for row in readiness if row["pair"] == "qwen3_1p7b")
    assert control["pair_pilot_ready"] == 0
    assert control["pilot_rank"] is None


def test_m3_m4_cannot_change_readiness() -> None:
    rows: list[dict[str, str]] = []
    for pair in audit.PAIR_ORDER:
        for task in audit.TASK_ORDER:
            block = _group(pair, task, 31 if pair == "tinyllama" else 0, 100)
            for index, row in enumerate(block):
                row["has_high_cardinality_m3"] = str(index % 2)
                row["has_strict_m4"] = str(index % 3 == 0)
            rows.extend(block)
    _, status, ranking, selected = audit.derive_readiness(rows)
    assert status == "DIAGNOSTIC_ONLY"  # pooled tinyllama count is 93, below 100
    assert ranking == []
    assert selected is None


def test_wilson_interval_and_bonferroni_lcb_are_finite() -> None:
    low, high = audit.wilson_interval(30, 100)
    lcb = audit.one_sided_wilson_lcb(30, 100, 0.05 / 9.0)
    assert low is not None and high is not None and lcb is not None
    assert 0.0 <= lcb <= low <= 0.30 <= high <= 1.0


def test_output_schema_is_exact_and_pair_task_count_is_60() -> None:
    assert audit.PAIR_TASK_ROWS == 60
    assert audit.PARENT_COLUMNS[0] == "schema_version"
    assert audit.PAIR_TASK_COLUMNS[-1] == "strict_support_ceiling"
    assert len(set(audit.PARENT_COLUMNS)) == len(audit.PARENT_COLUMNS)


def test_script_contains_no_automodel_or_cuda_call() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "AutoModel" not in source
    assert ".cuda(" not in source
    assert "torch.cuda" not in source
