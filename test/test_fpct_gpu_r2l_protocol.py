from __future__ import annotations

import json
from pathlib import Path

from script.analysis.fpct_gpu_r2l_protocol_verify import verify


ROOT = Path(__file__).resolve().parents[1]


def test_r2l_truth_table_fails_unknown_sidecar_closed() -> None:
    payload = json.loads(
        (ROOT / "recipe/eval_recipe/fpct_gpu_r2l/semantic_map_truth_table.json").read_text()
    )
    cases = {row["id"]: row for row in payload["cases"]}
    assert cases["native_outside_coverage"]["expected_equivalent"] is True
    assert cases["covered_metadata_missing"]["expected_equivalent"] is False


def test_r2l_protocol_diff_is_inside_function_allowlist() -> None:
    payload = verify(ROOT)
    assert payload["status"] == "GO"
    assert payload["truth_table_case_count"] == 7
