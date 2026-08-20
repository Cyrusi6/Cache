#!/usr/bin/env python3
"""Deterministic analysis and single-factor selection for E1-FAST-RCA."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.experiment.fpct_e1_fast_rca import (
    PROTOCOL_ID,
    TASKS,
    canonical_json_bytes,
    atomic_write,
    sha256_file,
)


BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260820
CONTROL_ATOL = 2e-5
SELECTABLE = (
    "lambda_025",
    "lambda_05",
    "parent_mass_preserving",
    "partition_composition",
)
TIE_ORDER = {name: index for index, name in enumerate(SELECTABLE)}
ROOT_CAUSE = {
    "lambda_025": "CANDIDATE_DEVIATION_SCALE_OR_PROJECTOR_CONTRACTION",
    "lambda_05": "CANDIDATE_DEVIATION_SCALE_OR_PROJECTOR_CONTRACTION",
    "parent_mass_preserving": "JENSEN_PARENT_EVIDENCE_INFLATION",
    "partition_composition": "PARTITION_AS_COMPETING_CANDIDATES",
}


def load_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for shard in plan["shards"]:
        path = Path(plan["output_root"]) / shard["output_relative"] / "sample_metrics.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if value.get("protocol_id") != PROTOCOL_ID:
                raise ValueError("row protocol mismatch")
            if not math.isfinite(float(value["gold_logp_mean"])):
                raise ValueError("nonfinite gold log-probability")
            rows.append(value)
    return rows


def matched_deltas(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    metadata: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            row["checkpoint_arm"],
            row["task"],
            row["content_group_sha256"],
        )
        variant = str(row["variant"])
        if variant in cells[key]:
            raise ValueError("duplicate matched intervention cell")
        cells[key][variant] = float(row["gold_logp_mean"])
        metadata[key] = {
            "seed": key[0],
            "checkpoint_arm": key[1],
            "task": key[2],
            "content_group_sha256": key[3],
        }
    result = []
    required = {
        "c_post", "f", "lambda_0", "lambda_025", "lambda_05", "lambda_2",
        "k_only_v_collapse", "parent_mass_preserving", "partition_composition",
    }
    for key, values in cells.items():
        if set(values) != required:
            raise ValueError(f"incomplete matched cell: {key}")
        row = {**metadata[key]}
        row["lambda0_minus_cpost"] = values["lambda_0"] - values["c_post"]
        row["f_minus_cpost"] = values["f"] - values["c_post"]
        for variant in required - {"f", "c_post"}:
            row[f"{variant}_minus_f"] = values[variant] - values["f"]
        result.append(row)
    return result


def _task_macro(values: Iterable[Mapping[str, Any]], field: str) -> float:
    by_task = defaultdict(list)
    for row in values:
        by_task[row["task"]].append(float(row[field]))
    if set(by_task) != set(TASKS):
        raise ValueError("task macro is incomplete")
    return mean(mean(by_task[task]) for task in TASKS)


def hierarchical_lcb(deltas: list[dict[str, Any]], field: str) -> tuple[float, float, float]:
    by_seed_task: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    seeds = sorted({int(row["seed"]) for row in deltas})
    for row in deltas:
        by_seed_task[(int(row["seed"]), row["task"])].append(row)
    if len(seeds) != 3:
        raise ValueError("hierarchical bootstrap requires three training seeds")
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        task_values = []
        for task in TASKS:
            groups = sorted(
                {row["content_group_sha256"] for row in by_seed_task[(seeds[0], task)]}
            )
            sampled_groups = [rng.choice(groups) for _ in groups]
            lookup = defaultdict(list)
            for seed in sampled_seeds:
                for row in by_seed_task[(seed, task)]:
                    lookup[row["content_group_sha256"]].append(float(row[field]))
            task_values.append(
                mean(mean(lookup[group]) for group in sampled_groups)
            )
        draws.append(mean(task_values))
    draws.sort()
    lower = draws[int(0.025 * BOOTSTRAP_REPLICATES)]
    upper = draws[min(BOOTSTRAP_REPLICATES - 1, int(0.975 * BOOTSTRAP_REPLICATES))]
    return lower, mean(draws), upper


def analyze(plan: Mapping[str, Any]) -> dict[str, Any]:
    rows = load_rows(plan)
    deltas = matched_deltas(rows)
    control_max = max(abs(float(row["lambda0_minus_cpost"])) for row in deltas)
    controls_go = control_max <= CONTROL_ATOL
    candidates = []
    for variant in SELECTABLE:
        field = f"{variant}_minus_f"
        task_means = {
            task: mean(float(row[field]) for row in deltas if row["task"] == task)
            for task in TASKS
        }
        arm_directions = {}
        for seed in sorted({int(row["seed"]) for row in deltas}):
            for arm in ("c_post_trained", "f_trained"):
                subset = [
                    row for row in deltas
                    if int(row["seed"]) == seed and row["checkpoint_arm"] == arm
                ]
                arm_directions[f"{seed}-{arm}"] = _task_macro(subset, field)
        positive_arms = sum(value > 0 for value in arm_directions.values())
        lower, bootstrap_mean, upper = hierarchical_lcb(deltas, field)
        task_macro_mean = _task_macro(deltas, field)
        eligible = (
            controls_go
            and lower > 0
            and positive_arms >= 5
            and all(value >= 0 for value in task_means.values())
        )
        candidates.append(
            {
                "variant": variant,
                "root_cause": ROOT_CAUSE[variant],
                "task_macro_mean_delta_logp": task_macro_mean,
                "task_means": task_means,
                "checkpoint_arm_directions": arm_directions,
                "positive_checkpoint_arms": positive_arms,
                "bootstrap_95_ci": [lower, upper],
                "bootstrap_mean": bootstrap_mean,
                "eligible": eligible,
            }
        )
    eligible = [value for value in candidates if value["eligible"]]
    winner = None
    if eligible:
        best_mean = max(
            float(value["task_macro_mean_delta_logp"]) for value in eligible
        )
        tied = [
            value for value in eligible
            if best_mean - float(value["task_macro_mean_delta_logp"]) <= 1e-6
        ]
        winner = min(tied, key=lambda value: TIE_ORDER[value["variant"]])
    diagnostic = {}
    for variant in ("k_only_v_collapse", "lambda_2"):
        field = f"{variant}_minus_f"
        diagnostic[variant] = {
            "task_macro_mean_delta_logp": _task_macro(deltas, field),
            "task_means": {
                task: mean(float(row[field]) for row in deltas if row["task"] == task)
                for task in TASKS
            },
            "selectable": False,
        }
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": (
            "SINGLE_FACTOR_SELECTED" if winner is not None
            else "NO_EXPLOITABLE_FIXED_CHECKPOINT_HEADROOM"
        ),
        "execution_sha": plan["execution_sha"],
        "row_count": len(rows),
        "matched_group_cells": len(deltas),
        "controls": {
            "lambda0_equals_cpost_atol": CONTROL_ATOL,
            "maximum_absolute_logp_delta": control_max,
            "go": controls_go,
        },
        "candidates": candidates,
        "diagnostic_only": diagnostic,
        "selected_variant": winner["variant"] if winner else None,
        "root_cause": winner["root_cause"] if winner else None,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "top_level_unit": "three_training_seeds_with_two_checkpoint_arms_paired",
        },
        "rope_frame_correction": {
            "status": "STRUCTURALLY_UNAVAILABLE_AT_FIXED_CHECKPOINT",
            "selectable": False,
            "reason": (
                "C2CProjector concatenates source and receiver K before its first "
                "nonlinear map; source/receiver head dimensions differ, so the "
                "frozen checkpoint exposes no separable P_K for exact R_r P_K R_s^-1."
            ),
        },
        "training_authorized": winner is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    result = analyze(plan)
    atomic_write(args.output, canonical_json_bytes(result))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
