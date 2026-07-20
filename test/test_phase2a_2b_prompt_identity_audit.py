from __future__ import annotations

from script.analysis.phase2a_2b_prompt_identity_audit import (
    PHASE2A0_MANIFEST,
    _fixed_environment_probe,
    _load_json,
    _source_artifact_paths,
)


def test_phase2a0_provenance_recovers_nine_llama_b6_artifacts() -> None:
    paths = _source_artifact_paths(_load_json(PHASE2A0_MANIFEST))
    assert len(paths) == 9
    assert all("/llama32_1b/b6/" in str(path) for path in paths)


def test_fixed_date_environment_matrix_is_identical() -> None:
    result = _fixed_environment_probe()
    assert result["all_hashes_and_counts_identical"] is True
    assert {row["timezone"] for row in result["cases"]} == {
        "UTC",
        "Asia/Shanghai",
        "America/New_York",
    }
    assert {row["locale"] for row in result["cases"]} == {
        "C",
        "zh_CN.utf8",
        "en_US.utf8",
    }
