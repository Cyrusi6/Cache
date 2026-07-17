#!/usr/bin/env python3
"""Deterministically shard independent Route-1 phase-1 runs across workers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


PAIR_WEIGHTS = {
    "tinyllama": 1.0,
    "qwen25_0p5b": 0.95,
    "qwen3_1p7b": 1.35,
    "llama32_1b": 1.1,
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_weight(run: dict[str, Any]) -> float:
    weight = PAIR_WEIGHTS.get(str(run.get("pair")), 1.0)
    if bool(run.get("gate_diagnostics", {}).get("required", False)):
        weight += 0.15
    return weight


def shard_plans(
    plan_paths: Sequence[Path],
    *,
    output_dir: Path,
    state_dir: Path,
    shard_count: int,
    lane_prefix: str,
    exclude_run_ids: Sequence[str] = (),
) -> dict[str, Any]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    plans = [_read_json(path.resolve()) for path in plan_paths]
    if not plans:
        raise ValueError("at least one source plan is required")
    if any(plan.get("phase") != "phase1" for plan in plans):
        raise ValueError("only phase1 plans can be sharded")
    suites = {str(plan.get("suite")) for plan in plans}
    if len(suites) != 1:
        raise ValueError(f"source plans disagree about suite: {sorted(suites)}")

    completed = {path.stem for path in (state_dir / "completed").glob("*.json")}
    explicitly_excluded = {str(run_id) for run_id in exclude_run_ids}
    candidates: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()
    ordinal = 0
    for plan in plans:
        for raw_run in plan.get("runs", []):
            run = copy.deepcopy(raw_run)
            run_id = str(run["run_id"])
            if run_id in seen:
                raise ValueError(f"duplicate run across source plans: {run_id}")
            seen.add(run_id)
            if run_id not in completed and run_id not in explicitly_excluded:
                run["depends_on_runs"] = []
                candidates.append((ordinal, run))
            ordinal += 1

    buckets: list[list[tuple[int, dict[str, Any]]]] = [
        [] for _ in range(shard_count)
    ]
    loads = [0.0 for _ in range(shard_count)]
    for item in sorted(
        candidates,
        key=lambda value: (-_run_weight(value[1]), value[0], value[1]["run_id"]),
    ):
        index = min(range(shard_count), key=lambda candidate: (loads[candidate], candidate))
        buckets[index].append(item)
        loads[index] += _run_weight(item[1])

    output_dir = output_dir.resolve()
    shard_records = []
    for index, bucket in enumerate(buckets, start=1):
        lane = f"{lane_prefix}_{index}"
        runs = [run for _ordinal, run in sorted(bucket, key=lambda value: value[0])]
        plan = {
            "schema_version": 1,
            "suite": next(iter(suites)),
            "lane": lane,
            "phase": "phase1",
            "hardware": {"profile": "assigned_by_two_gpu_adapter"},
            "state_dir": str(state_dir.resolve()),
            "gate_contract": {
                "reproduction": "pass|pending|fail",
                "conditional": "pass|pending|fail",
            },
            "runs": runs,
        }
        plan_path = output_dir / f"{lane}.phase1.json"
        _write_json(plan_path, plan)
        shard_records.append(
            {
                "lane": lane,
                "plan": str(plan_path),
                "plan_sha256": _sha256(plan_path),
                "estimated_weight": loads[index - 1],
                "run_ids": [str(run["run_id"]) for run in runs],
            }
        )

    manifest = {
        "schema_version": 1,
        "source_plans": [
            {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
            for path in plan_paths
        ],
        "state_dir": str(state_dir.resolve()),
        "completed_runs_excluded": sorted(completed & seen),
        "reserved_runs_excluded": sorted(explicitly_excluded & seen),
        "pending_run_count": len(candidates),
        "shards": shard_records,
    }
    _write_json(output_dir / "shard_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--lane-prefix", default="lane_bc_shard")
    parser.add_argument("--exclude-run", action="append", default=[])
    args = parser.parse_args()
    manifest = shard_plans(
        args.plan,
        output_dir=args.output_dir,
        state_dir=args.state_dir,
        shard_count=args.shards,
        lane_prefix=args.lane_prefix,
        exclude_run_ids=args.exclude_run,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
