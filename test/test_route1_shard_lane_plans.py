from __future__ import annotations

import json
from pathlib import Path

from script.k8s import route1_shard_lane_plans as shard


def _write_plan(path: Path, lane: str, runs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite": "route1_v22_identifiability",
                "lane": lane,
                "phase": "phase1",
                "state_dir": "state",
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )


def test_shard_plans_excludes_completed_and_removes_serial_dependencies(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    (state / "completed").mkdir(parents=True)
    (state / "completed/done.json").write_text("{}\n", encoding="utf-8")
    plan_b = tmp_path / "lane_b.json"
    plan_c = tmp_path / "lane_c.json"
    _write_plan(
        plan_b,
        "lane_b",
        [
            {
                "run_id": "done",
                "pair": "tinyllama",
                "depends_on_runs": [],
                "gate_diagnostics": {"required": False},
            },
            {
                "run_id": "heavy",
                "pair": "qwen3_1p7b",
                "depends_on_runs": ["done"],
                "gate_diagnostics": {"required": True},
            },
        ],
    )
    _write_plan(
        plan_c,
        "lane_c",
        [
            {
                "run_id": "light-a",
                "pair": "qwen25_0p5b",
                "depends_on_runs": ["heavy"],
                "gate_diagnostics": {"required": False},
            },
            {
                "run_id": "light-b",
                "pair": "tinyllama",
                "depends_on_runs": ["light-a"],
                "gate_diagnostics": {"required": False},
            },
        ],
    )

    output = tmp_path / "shards"
    manifest = shard.shard_plans(
        [plan_b, plan_c],
        output_dir=output,
        state_dir=state,
        shard_count=2,
        lane_prefix="worker",
        exclude_run_ids=["light-b"],
    )

    assert manifest["pending_run_count"] == 2
    assert manifest["completed_runs_excluded"] == ["done"]
    assert manifest["reserved_runs_excluded"] == ["light-b"]
    assigned = []
    for record in manifest["shards"]:
        plan = json.loads(Path(record["plan"]).read_text(encoding="utf-8"))
        assigned.extend(run["run_id"] for run in plan["runs"])
        assert all(not run["depends_on_runs"] for run in plan["runs"])
    assert sorted(assigned) == ["heavy", "light-a"]
    assert len(assigned) == len(set(assigned))
